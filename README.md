# OCR 테스트 수정 중

## 설치 방법

### 1. 필수 패키지 설치

#### macOS
```bash
# Homebrew를 통한 Tesseract OCR 설치
brew install tesseract
brew install tesseract-lang  # 한국어 언어 패키지 포함

# Python 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# Python 패키지 설치
uv pip install -r requirements.txt
```

#### Ubuntu/Debian
```bash
# Tesseract OCR 설치
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-kor  # 한국어 언어 패키지

# Python 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# Python 패키지 설치
uv pip install -r requirements.txt
```

### 2. 필요한 Python 패키지
- opencv-python
- numpy
- pytesseract
- easyocr
- pdf2image
- pytest

## 실행 방법

1. 가상환경 활성화
```bash
source .venv/bin/activate
```

2. 테스트 실행
```bash
pytest tests/test_markitdown.py -v
```

## 프로젝트 구조
```
ocr/
├── .venv/                  # 가상환경
├── markitdown/            # 패키지 디렉토리
├── tests/                 # 테스트 파일
│   └── test_markitdown.py
├── test_files/           # 테스트용 이미지/PDF 파일
├── output/               # OCR 결과 출력 디렉토리
├── requirements.txt      # Python 패키지 의존성
└── README.md            # 프로젝트 문서
```

## 주의사항
- Tesseract OCR이 시스템에 설치되어 있어야 합니다.
- 한국어 인식을 위해서는 Tesseract의 한국어 언어 패키지가 필요합니다.
- 이미지 파일은 `test_files` 디렉토리에 위치해야 합니다.
- OCR 결과는 `output` 디렉토리에 저장됩니다. 