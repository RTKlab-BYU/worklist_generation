import subprocess
import sys
import re
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


def validate_outputs(output_dir, twocol_2xp=False):
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

    for df, label in [(ms, "MS"), (lc, "LC")]:
        _validate_file_names(df, label, twocol_2xp)


def _validate_file_names(df, label, twocol_2xp):
    """Run all File Name column checks for a given dataframe."""
    # Find the File Name column (search across all columns since the file has
    # a metadata header row before the actual column header row)
    file_name_col = None
    for col in df.columns:
        if df[col].astype(str).str.contains("File Name", na=False).any():
            file_name_col = col
            break
    assert file_name_col is not None, f"{label}: Could not find 'File Name' column"

    # Get the index of the header row and extract file names below it
    header_row_idx = df[df[file_name_col].astype(str).str.contains("File Name", na=False)].index[0]
    file_names = df.loc[header_row_idx + 1:, file_name_col].dropna().astype(str).tolist()

    assert len(file_names) > 0, f"{label}: No file names found"

    # 1. At least one file name contains "lib" (case-insensitive)
    assert any("lib" in fn.lower() for fn in file_names), (
        f"{label}: No file names containing 'lib'"
    )

    # 2. At least one file name contains "qc" (case-insensitive)
    assert any("qc" in fn.lower() for fn in file_names), (
        f"{label}: No file names containing 'qc'"
    )

    # 3. Every block contains one of every condition.
    #    If this is a 2-column system with two experiments, split file names
    #    into two interleaved groups (every other) and check each separately.
    def check_block_completeness(fns, label_suffix=""):
        block_pattern = re.compile(r"_([^_]+)_(blo\d+)_", re.IGNORECASE)
        blocks: dict[str, list[str]] = {}
        for fn in fns:
            m = block_pattern.search(fn)
            if m:
                condition = m.group(1)
                block_id = m.group(2).lower()
                blocks.setdefault(block_id, []).append(condition)

        assert len(blocks) > 0, f"{label}{label_suffix}: No block file names found"
        print(len(blocks))

        all_conditions = set(c for conditions in blocks.values() for c in conditions)
        for block_id, conditions in blocks.items():
            missing = all_conditions - set(conditions)
            assert not missing, (
                f"{label}{label_suffix}: Block '{block_id}' is missing conditions: {missing}"
            )

    # NEW: Filter for block files BEFORE slicing to preserve experiment parity
    block_pattern = re.compile(r"_([^_]+)_(blo\d+)_", re.IGNORECASE)
    block_files = [fn for fn in file_names if block_pattern.search(fn)]

    if twocol_2xp:
        check_block_completeness(block_files[0::2], label_suffix=" (experiment 1)")
        check_block_completeness(block_files[1::2], label_suffix=" (experiment 2)")
    else:
        check_block_completeness(block_files)

    # 4. At least one file name contains "tb" or "trueblank" (case-insensitive)
    assert any("tb" in fn.lower() or "trueblank" in fn.lower() or "true_blank" in fn.lower()
               for fn in file_names), (
        f"{label}: No file names containing 'tb' or 'trueblank'"
    )

    # 5. At least one file name contains "sys" (case-insensitive)
    assert any("sys" in fn.lower() for fn in file_names), (
        f"{label}: No file names containing 'sys'"
    )

    # 6. All non-block files that appear between blocks (i.e. not pre/post)
    #    must be labeled "other" (case-insensitive).
    #    "Between blocks" means: not containing 'pre', 'post', or a block tag (blo\d+),
    #    and not being a lib or sys file (which have no positional label).
    inter_block_pattern = re.compile(r"_(pre|post|other|blo\d+|lib\d+|SysQC)_", re.IGNORECASE)
    for fn in file_names:
        m = inter_block_pattern.search(fn)
        if m is None:
            continue  # file has no positional label — skip
        positional_label = m.group(1).lower()
        # Files that are between blocks but are not pre/post/blo#/SysQC must be "other"
        if positional_label not in ("pre", "post", "sysqc", "other") and not positional_label.startswith("blo") and not positional_label.startswith("lib"):
            assert False, (
                f"{label}: Inter-block file '{fn}' has unexpected label "
                f"'{positional_label}' — expected 'other'"
            )


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
    validate_outputs(output_dir, True)

def test_stage3_two_column_two_experiments_different_lib(tmp_path):
    """
    2 column, 2 experiments, different library
    """
    worklist = Path("tests/data/2col_2xp_diflib.xlsx")
    output_dir = tmp_path

    run_stage3(worklist, output_dir)
    validate_outputs(output_dir, True)

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