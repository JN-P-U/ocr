import subprocess
import sys


def run_pytest():
    print("테스트 실행 중...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_markitdown.py", "-v"]
    )
    if result.returncode != 0:
        print("테스트 실패!")
        sys.exit(result.returncode)
    else:
        print("테스트 성공!")


if __name__ == "__main__":
    run_pytest()
