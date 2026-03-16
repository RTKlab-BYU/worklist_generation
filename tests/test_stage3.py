import subprocess
import sys
from pathlib import Path
import pandas as pd


RUN_SCRIPT = Path("run.py")


def run_stage3(worklist, output_dir):
    """Helper to run stage 3."""
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "-s", "3",
        "-w", str(worklist),
        "-o", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    return result


def validate_outputs(output_dir):
    """Check MS and LC files exist and look valid."""
    files = list(output_dir.glob("*.csv"))

    assert len(files) == 2, "Expected exactly 2 output CSV files"

    ms_file = [f for f in files if "MS" in f.name][0]
    lc_file = [f for f in files if "LC" in f.name][0]

    ms = pd.read_csv(ms_file, header=None)
    lc = pd.read_csv(lc_file, header=None)

    assert ms.shape[0] > 10
    assert lc.shape[0] > 10

    assert ms.shape[1] >= 5
    assert lc.shape[1] >= 5


def test_stage3_one_column_one_experiment(tmp_path):
    """
    1 column, 1 experiment
    """
    worklist = Path("tests/data/1col_1xp.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir)


def test_stage3_two_column_one_experiment(tmp_path):
    """
    2 column, 1 experiment
    """
    worklist = Path("tests/data/2col_1xp.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir)


def test_stage3_two_column_two_experiments_same_lib(tmp_path):
    """
    2 column, 2 experiments, same library
    """
    worklist = Path("tests/data/2col_2xp.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir)

def test_stage3_two_column_two_experiments_different_lib(tmp_path):
    """
    2 column, 2 experiments, different library
    """
    worklist = Path("tests/data/2col_2xp_diflib.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir)

def test_stage3_mock_b_and_t_exp(tmp_path):
    """
    1 column, 1 experiment, full mock B and T cell template from the paper
    """
    worklist = Path("tests/data/mock_b_and_t_cells_exp.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir)

def test_stage3_drug_treatment_exp(tmp_path):
    """
    1 column, 1 experiment, full mock drug treament template
    """
    worklist = Path("tests/data/mock_drug_treatment_exp.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir)