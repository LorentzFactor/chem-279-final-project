import pathlib
import subprocess
import shutil

import pytest

BASE_DIR = pathlib.Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "sample_input"
SAMPLE_OUTPUT_DIR = BASE_DIR / "sample_output"
STUDENT_OUTPUT_DIR = BASE_DIR / "student_output"
BINARY_DIR = BASE_DIR / "build"

@pytest.fixture(scope="session")
def sample_output_dir():
    return SAMPLE_OUTPUT_DIR

@pytest.fixture(scope="session")
def student_output_dir(): 
    return STUDENT_OUTPUT_DIR

def pytest_configure():
    run_all_test_cases()

def run_all_test_cases():
    if STUDENT_OUTPUT_DIR.exists():
        shutil.rmtree(STUDENT_OUTPUT_DIR)
    STUDENT_OUTPUT_DIR.mkdir(parents=True)
    
    for question_id in [f.name for f in BINARY_DIR.iterdir() if f.is_file() and (("pl" in f.name) or ("hw" in f.name))]:
        input_subdir = INPUT_DIR / question_id
        student_output_subdir = STUDENT_OUTPUT_DIR / question_id
        executable_path = BINARY_DIR / question_id

        student_output_subdir.mkdir(parents=True, exist_ok=True)
        for test_file in input_subdir.iterdir():
            if test_file.is_file():

                student_output_path = student_output_subdir / test_file.stem
                result = subprocess.run(args=[executable_path, test_file], capture_output=True, text=True, check=True)

                student_output_path.with_suffix(".stdout").write_text(result.stdout)