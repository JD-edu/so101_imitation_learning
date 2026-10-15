"""기존 실행 이름 호환용. 새 실행 파일은 207_ACT_CVAE_infer.py입니다."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("207_ACT_CVAE_infer.py")), run_name="__main__")
