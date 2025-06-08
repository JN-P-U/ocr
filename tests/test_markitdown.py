import json
import os
from collections import defaultdict

import cv2
import easyocr
import numpy as np
import pytesseract
import pytest
from markitdown._base_converter import DocumentConverterResult
from pdf2image import convert_from_path

from markitdown import MarkItDown


@pytest.fixture
def markitdown():
    return MarkItDown()


def test_convert_docx(markitdown):
    # 테스트용 docx 파일 경로
    test_file = "test_files/test.docx"

    # 테스트 파일이 존재하는지 확인
    assert os.path.exists(test_file), f"테스트 파일이 존재하지 않습니다: {test_file}"

    # 변환 실행
    result = markitdown.convert(test_file)

    # 결과가 DocumentConverterResult 타입인지 확인
    assert isinstance(result, DocumentConverterResult)

    # 결과의 markdown이 문자열인지 확인
    assert isinstance(result.markdown, str)

    # 결과가 비어있지 않은지 확인
    assert len(result.markdown) > 0


def test_convert_invalid_file(markitdown):
    # 존재하지 않는 파일에 대한 테스트
    with pytest.raises(Exception):
        markitdown.convert("nonexistent.docx")


def test_convert_empty_file(markitdown):
    # 빈 파일에 대한 테스트
    empty_file = "test_files/empty.docx"
    if os.path.exists(empty_file):
        result = markitdown.convert(empty_file)
        assert isinstance(result, DocumentConverterResult)
        assert isinstance(result.markdown, str)


def preprocess_image(image):
    """OCR 친화적인 전처리 파이프라인 (AdaptiveThreshold + BilateralFilter)."""
    # 1) 그레이스케일
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) 해상도가 낮을 경우 업스케일 (최대 변 1800px)
    h, w = gray.shape
    if max(h, w) < 1800:
        scale = 1800 / max(h, w)
        gray = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    # 3) 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3-b) Unsharp mask to sharpen strokes
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    # 4) 노이즈 감소 (bilateral)
    denoised = cv2.bilateralFilter(sharpened, 9, 75, 75)

    # 5) Adaptive Threshold (block 25, C 10) – 더 촘촘한 문자 보존
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        10,
    )

    # 6) 형태학적 closing → 끊긴 획 연결
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def is_table_structure(rows):
    """행과 열의 구조를 분석하여 표인지 판단"""
    if not rows:
        return False

    # 각 행의 열 개수 확인
    col_counts = [len(row) for row in rows]

    # 모든 행의 열 개수가 2개 이상이고, 대부분의 행이 같은 열 개수를 가지는지 확인
    if len(set(col_counts)) <= 2 and min(col_counts) >= 2:
        return True

    return False


def test_convert_pdf_file(markitdown):
    # 테스트용 PDF 파일 경로
    test_pdf = "test_files/test_pdf.pdf"

    if os.path.exists(test_pdf):
        # PDF를 이미지로 변환
        images = convert_from_path(test_pdf, dpi=400, fmt="png")

        # Tesseract OCR 설정
        custom_config = r"--oem 1 --psm 4 -l kor+eng --dpi 400"

        for i, image in enumerate(images):
            # 이미지를 numpy 배열로 변환
            image_np = np.array(image)

            # 이미지 전처리
            processed_image = preprocess_image(image_np)

            # OCR 수행
            text = pytesseract.image_to_string(processed_image, config=custom_config)

            # 결과를 파일로 저장
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            # PDF 파일명에서 확장자를 제외한 이름을 가져옴
            base_name = os.path.splitext(os.path.basename(test_pdf))[0]

            cleaned_lines = _postprocess_lines(text.splitlines())
            text = "\n".join(cleaned_lines)

            # 텍스트 파일로 저장
            text_output_file = os.path.join(output_dir, f"{base_name}_page{i+1}.txt")
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"\n페이지 {i+1}의 텍스트가 저장되었습니다: {text_output_file}")
            print("\n추출된 텍스트:")
            print(text)


def clean_text(text):
    """인식된 텍스트를 정리하는 함수"""
    # 숫자와 한글 사이의 공백 제거
    text = text.replace(" ", "")

    # 특수문자 처리
    text = text.replace("?", "? ")
    text = text.replace("!", "! ")

    # 연속된 공백 제거
    text = " ".join(text.split())

    return text


def is_number(text):
    """텍스트가 숫자인지 확인하는 함수"""
    return text.strip().isdigit()


def merge_text_by_position(texts, positions, threshold=10):
    """위치 기반으로 텍스트를 병합하는 함수"""
    if not texts:
        return []

    # y좌표 기준으로 그룹화
    y_groups = defaultdict(list)
    for text, pos in zip(texts, positions):
        y_coord = np.mean([p[1] for p in pos])
        y_groups[y_coord].append((text, pos))

    # 각 그룹 내에서 x좌표 기준으로 정렬하고 텍스트 병합
    merged_texts = []
    for y_coord in sorted(y_groups.keys()):
        group = y_groups[y_coord]
        group.sort(key=lambda x: np.mean([p[0] for p in x[1]]))

        # 같은 줄의 텍스트들을 하나로 병합
        line_parts = []
        current_text = ""
        current_number = ""

        for text, pos in group:
            # 숫자만 있는 경우
            if text.strip().isdigit():
                if current_text:
                    line_parts.append(current_text)
                    current_text = ""
                current_number = text
            # 한글이나 특수문자가 포함된 경우
            else:
                if current_number:
                    line_parts.append(current_number)
                    current_number = ""
                if current_text:
                    current_text += text
                else:
                    current_text = text

        # 남은 텍스트 처리
        if current_number:
            line_parts.append(current_number)
        if current_text:
            line_parts.append(current_text)

        # 줄의 모든 부분을 하나로 합치기
        line_text = " ".join(line_parts)
        if line_text:
            merged_texts.append(line_text)

    return merged_texts


def get_easyocr_result(image):
    """EasyOCR로 텍스트 추출"""
    reader = easyocr.Reader(
        ["ko", "en"],
        gpu=False,
        model_storage_directory=os.path.abspath("."),
        download_enabled=True,  # 자동 모델 다운로드 허용
        recog_network="korean_g2",  # 공식 지원 모델
    )
    results = reader.readtext(
        image,
        detail=1,
        allowlist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ가-힣ㄱ-ㅎㅏ-ㅣ.,!?@#$%^&*()_+-=[]{}|;:\"'<>/\\~` ",
    )
    return [(bbox, text, prob) for bbox, text, prob in results if prob > 0.2]


def get_tesseract_result(image):
    """Tesseract OCR로 텍스트 추출"""
    # 1차 시도: 전처리 이미지를 그대로 사용
    primary_config = r"--oem 1 --psm 4 -l kor+eng --dpi 400"
    text = pytesseract.image_to_string(image, config=primary_config)

    # 2차 시도: 결과가 비어 있으면 원본 또는 grayscale 변환본으로 재시도
    if not text.strip():
        # 이미 grayscale(또는 binary) 이미지일 수도 있으므로 채널 개수를 확인
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            gray = image  # 이미 1채널이면 그대로 사용
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fallback_config = r"--oem 1 --psm 3 -l kor+eng --dpi 400"
        text = pytesseract.image_to_string(gray, config=fallback_config)

    return text


# --- HELPER: 한글 single-char spacing 결합 함수 ---
def _collapse_korean_spacing(line: str) -> str:
    """
    연속된 한글 1글자 토큰들 사이의 공백을 제거하여 단어로 결합한다.
    예) '인 간 집 사' -> '인간 집사'
    """

    def is_korean_char(token: str) -> bool:
        return all("\uac00" <= ch <= "\ud7a3" for ch in token)

    tokens = line.split()
    merged_tokens = []
    buffer = ""

    for tok in tokens:
        if len(tok) == 1 and is_korean_char(tok):
            buffer += tok
        else:
            if buffer:
                merged_tokens.append(buffer)
                buffer = ""
            merged_tokens.append(tok)

    if buffer:
        merged_tokens.append(buffer)

    return " ".join(merged_tokens)


def _postprocess_lines(lines):
    """OCR 후 라인별 후처리: 불필요 숫자 제거 + 한글 single-char spacing 결합."""
    processed = []
    for ln in lines:
        # 페이지 넘버 등 4자리 이하의 순수 숫자 라인은 제거
        stripped = ln.strip()
        if stripped.isdigit() and len(stripped) <= 4:
            continue
        merged = _collapse_korean_spacing(stripped)
        if merged:
            processed.append(merged)
    return processed


def merge_results(easyocr_results, tesseract_text):
    """여러 OCR 결과를 병합하고 최적의 결과 선택"""
    # EasyOCR 결과 처리
    easyocr_texts = []
    for _, text, _ in easyocr_results:
        easyocr_texts.append(text)

    # Tesseract 결과 처리
    tesseract_lines = [
        line.strip() for line in tesseract_text.split("\n") if line.strip()
    ]

    # 결과 비교 및 선택
    final_texts = []

    # 각 엔진의 결과를 비교하여 가장 긴 텍스트 선택
    max_lines = max(len(easyocr_texts), len(tesseract_lines))

    for i in range(max_lines):
        texts = []
        if i < len(easyocr_texts):
            texts.append(easyocr_texts[i])
        if i < len(tesseract_lines):
            texts.append(tesseract_lines[i])

        if texts:
            # 가장 긴 텍스트 선택 (일반적으로 더 많은 정보를 포함)
            selected_text = max(texts, key=len)
            final_texts.append(selected_text)

    # 후처리: 공백/숫자 라인 정리
    final_texts = _postprocess_lines(final_texts)
    return final_texts


def test_convert_image(markitdown):
    # 테스트용 이미지 파일 경로
    test_image = "test_files/논리.png"

    # 테스트 파일이 존재하는지 확인
    assert os.path.exists(
        test_image
    ), f"테스트 이미지가 존재하지 않습니다: {test_image}"

    # 이미지 읽기
    image = cv2.imread(test_image)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {test_image}")

    # 이미지 전처리
    processed_image = preprocess_image(image)

    # 각 OCR 엔진으로 텍스트 추출
    easyocr_results = get_easyocr_result(processed_image)
    tesseract_text = get_tesseract_result(processed_image)

    # 결과 병합
    final_texts = merge_results(easyocr_results, tesseract_text)

    # 결과를 파일로 저장
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 파일명에서 확장자를 제외한 이름을 가져옴
    base_name = os.path.splitext(os.path.basename(test_image))[0]

    # 텍스트 파일로 저장
    text_output_file = os.path.join(output_dir, f"{base_name}.txt")
    with open(text_output_file, "w", encoding="utf-8") as f:
        for text in final_texts:
            f.write(text + "\n")

    print(f"\n텍스트가 저장되었습니다: {text_output_file}")
    print("\n추출된 텍스트:")
    for text in final_texts:
        print(text)
