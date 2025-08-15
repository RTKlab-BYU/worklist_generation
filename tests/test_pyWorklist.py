import os
import sys
import math
import csv
import types
import importlib.util
import pandas as pd
import pytest
from pathlib import Path

# -----------------------------
# Path auto-discovery
# -----------------------------

# Determine base project dir (folder containing pyWorklist.py)
this_dir = Path(__file__).resolve().parent
candidate_dirs = [this_dir, this_dir.parent]

py_path = None
xlsx_path = None
for d in candidate_dirs:
    py_candidate = d / "pyWorklist.py"
    xlsx_candidate = d / "worklist.xlsx"
    if py_candidate.exists():
        py_path = str(py_candidate)
    if xlsx_candidate.exists():
        xlsx_path = str(xlsx_candidate)

if not py_path:
    raise FileNotFoundError("Could not find pyWorklist.py in test folder or parent.")
if not xlsx_path:
    print("⚠️  worklist.xlsx not found — Excel-based tests will be skipped.")

# Set env vars so our loader uses them
os.environ["PYWORKLIST_PATH"] = py_path
if xlsx_path:
    os.environ["WORKLIST_XLSX_PATH"] = xlsx_path

# -----------------------------
# Helpers & fixtures
# -----------------------------

def load_module():
    """Import pyWorklist dynamically from the detected path."""
    module_path = os.getenv("PYWORKLIST_PATH")
    spec = importlib.util.spec_from_file_location("pyWorklist", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope="session")
def worklist_df():
    """Load Excel if present, otherwise skip Excel-based tests."""
    xlsx_path = os.getenv("WORKLIST_XLSX_PATH")
    if not xlsx_path or not Path(xlsx_path).exists():
        pytest.skip("Excel file not found; skipping Excel-based tests.")
    return pd.read_excel(xlsx_path)

@pytest.fixture(autouse=True)
def _reseed():
    import random
    random.seed(12345)


# -----------------------------
# Basic function tests
# -----------------------------

def test_generate_seed_deterministic_print(capsys):
    mod = load_module()
    mod.generate_seed(42)
    captured = capsys.readouterr().out.strip()
    assert captured == "42"

def test_read_excel_to_df_success(worklist_df):
    mod = load_module()
    # Should return a DataFrame with at least 1 column and row
    df = mod.read_excel_to_df(os.getenv("WORKLIST_XLSX_PATH", "/mnt/data/worklist.xlsx"))
    assert hasattr(df, "shape") and df.shape[0] > 0 and df.shape[1] > 0

def test_read_excel_to_df_missing_raises(tmp_path):
    mod = load_module()
    missing = tmp_path / "missing.xlsx"
    with pytest.raises(ValueError) as ei:
        mod.read_excel_to_df(str(missing))
    assert "not found" in str(ei.value)

def test_validate_and_convert_spacing_ok():
    mod = load_module()
    assert mod.validate_and_convert_spacing([1, 2.0, 3], "qc_spacing") == [1,2,3]

def test_validate_and_convert_spacing_rejects_non_ints():
    mod = load_module()
    with pytest.raises(ValueError):
        mod.validate_and_convert_spacing([1, 2.2, 3], "qc_spacing")

# -----------------------------
# Integration with provided sheet
# -----------------------------

def test_condition_dict_keys_and_labels(worklist_df):
    mod = load_module()
    conds = mod.condition_dict(worklist_df)
    # Expect at least several conditions including 1..5 for this sheet
    assert 1 in conds and 2 in conds and 3 in conds
    # Sanity-check some labels
    assert isinstance(conds[1][0], str)
    assert isinstance(conds[2][0], str)

def test_separate_plates_structure(worklist_df):
    mod = load_module()
    nbcode, pathway, lc_number, wet_amounts, plates, num_to_run = mod.separate_plates(worklist_df)
    # From the provided sheet we expect 3 plates
    assert set(plates.keys()) == {"R_redplate", "B_blueplate", "G_greenplate"}
    # And a 1-column system per this example
    assert lc_number in (1,2)
    assert isinstance(wet_amounts, dict) and len(wet_amounts) > 0
    assert isinstance(num_to_run, dict)

def test_spacing_tuple_contents(worklist_df):
    mod = load_module()
    spacing_tuple = mod.spacing(worklist_df)
    # spacing() returns an 8-tuple with the first element a list of 4 spacing triplets
    assert isinstance(spacing_tuple, tuple) and len(spacing_tuple) == 8
    spacings, Lib_placement, SysValid_interval, experiment1, experiment2, lib_same, two_xp_TB, even = spacing_tuple
    assert isinstance(spacings, list) and len(spacings) == 4
    for tri in spacings:
        assert isinstance(tri, list) and len(tri) == 3 and all(isinstance(x, int) for x in tri)
    assert Lib_placement in ("Before", "After")
    assert isinstance(SysValid_interval, (int, float))
    assert lib_same in ("Yes", "No")
    assert even in ("Yes", "No")

def test_process_plate_and_wells_format(worklist_df):
    mod = load_module()
    nbcode, pathway, lc_number, wet_amounts, plates, num_to_run = mod.separate_plates(worklist_df)
    # Pick one plate
    name, plate_df = next(iter(plates.items()))
    wells = mod.process_plate(plate_df, name, wet_amounts)
    # Each well entry should be [condition_number, position_string]
    assert all(isinstance(w, list) and len(w) == 2 and isinstance(w[0], (int, float)) and isinstance(w[1], str) for w in wells)

def test_compare_wells_and_counts_detects_mismatch(worklist_df):
    mod = load_module()
    nbcode, pathway, lc_number, wet_amounts, plates, num_to_run = mod.separate_plates(worklist_df)
    spacings, Lib_placement, SysValid_interval, experiment1, experiment2, lib_same, two_xp_TB, even = mod.spacing(worklist_df)
    name, plate_df = next(iter(plates.items()))
    wells = mod.process_plate(plate_df, name, wet_amounts)
    # We expect the raw wells from the sheet not to include all required QCs/Blanks as per spacings[0],
    # so the validator should raise.
    with pytest.raises(ValueError):
        mod.compare_wells_and_counts(wells, mod.condition_dict(worklist_df), spacings, wet_amounts=None)

# -----------------------------
# Mid-pipeline tests (filenames & methods)
# -----------------------------

def test_create_instrument_methods_adds_channel_prefixes():
    mod = load_module()
    # Simulate two LC method path/name pairs
    lc_number = 2
    methodpaths = [r"C:\\LC\\Methods", r"C:\\LC\\Methods"]
    methods = ["GradientX", "GradientY"]
    out = mod.create_instrument_methods(lc_number, methodpaths[:], methods[:], csv_file="LC")
    # Expect ChannelA_ on first, ChannelB_ on second
    assert out[0].endswith("ChannelA_GradientX")
    assert out[1].endswith("ChannelB_GradientY")

def test_create_filenames_happy_path():
    mod = load_module()
    lc_number = 1
    conditions = {
        1: ["Lib", "lib", "dp", "mp", "dry", "msm", "ldp", "lmp", "wet", "lcm"],
        2: ["QC", "qc", "dp", "mp", "dry", "msm", "ldp", "lmp", "wet", "lcm"],
    }
    nbcode = "NB001"
    well_conditions = [1,2,1]
    block_runs = ["blo1_run1", "blo1_run2", "blo2_run1"]
    positions = ["RA1", "RB1", "RC1"]
    reps = [1,1,2]
    msmethods = ["dry", "dry", "dry"]
    filenames = mod.create_filenames(lc_number, conditions, nbcode, well_conditions, block_runs, positions, reps, msmethods)
    assert filenames[0].startswith("NB001_lib_blo1_run1_RA1_rep1")

def test_create_csv_to_send_writes_file(tmp_path):
    mod = load_module()
    # Temporarily redirect output folder
    mod.output_folder = str(tmp_path)
    conditions = {
        1: ["Lib", "lib", "C:\\\\MSDATA", "C:\\\\MSMETHODS", "dry", "MSM1", "C:\\\\LCDATA", "C:\\\\LCMETHODS", "wet", "LCM1"],
        2: ["QC", "qc", "C:\\\\MSDATA", "C:\\\\MSMETHODS", "dry", "MSM2", "C:\\\\LCDATA", "C:\\\\LCMETHODS", "wet", "LCM2"],
    }
    nbcode = "NB001"
    lc_number = 1
    blank_method = "BlankMeth"
    sample_type = "Unknown"
    filenames = ["NB001_qc_blo1_run1_RA1_rep1"]
    well_conditions = [2]
    positions = ["RA1"]
    inj_vol = 1
    out = mod.create_csv_to_send("MS", conditions, nbcode, lc_number, blank_method, sample_type, filenames, well_conditions, positions, inj_vol)
    assert os.path.exists(out)
    # Inspect that header rows exist and a data row is written
    with open(out, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0].startswith("Bracket Type=")
    assert rows[1] == ["Sample Type", "File Name", "Path", "Instrument Method", "Position", "Inj Vol"]
    assert rows[-1][0] == sample_type
    assert filenames[0] in rows[-1][1]

# -----------------------------
# Small utilities
# -----------------------------

def test_fully_flatten_and_rep_tracker():
    mod = load_module()
    # Create a toy, three-layer nested list: [[[cond, pos, block_run]]...]
    non_flat = [
        [[[1, "RA1", "blo1_run1"]], [[1, "RA2", "blo1_run2"]]],
        [[[2, "RB1", "blo2_run1"]]],
    ]
    flat = mod.flattener(non_flat)
    assert flat == [
        [1, "RA1", "blo1_run1"],
        [1, "RA2", "blo1_run2"],
        [2, "RB1", "blo2_run1"],
    ]

# -----------------------------
# Additional edge case tests
# -----------------------------

def test_validate_and_convert_spacing_empty_list():
    mod = load_module()
    # Should accept empty list and return empty list
    assert mod.validate_and_convert_spacing([], "qc_spacing") == []

def test_validate_and_convert_spacing_negative_values():
    mod = load_module()
    # Should still convert negative ints fine
    vals = [-1, -2.0, 3]
    assert mod.validate_and_convert_spacing(vals, "qc_spacing") == [-1, -2, 3]

def test_read_excel_to_df_with_non_excel_extension(tmp_path):
    mod = load_module()
    # Create a CSV file and try reading it (should fail)
    fakefile = tmp_path / "fake.txt"
    fakefile.write_text("Not an excel file")
    with pytest.raises(ValueError):
        mod.read_excel_to_df(str(fakefile))

def test_create_instrument_methods_with_lc_number_one():
    mod = load_module()
    lc_number = 1
    methodpaths = [r"C:\\LC\\Methods"]
    methods = ["GradientZ"]
    out = mod.create_instrument_methods(lc_number, methodpaths[:], methods[:], csv_file="LC")
    # Only one method, no channel prefix
    assert out[0].endswith("GradientZ")

def test_create_filenames_with_special_characters():
    mod = load_module()
    lc_number = 1
    conditions = {1: ["Lib", "lib", "", "", "", "", "", "", "", ""]}
    nbcode = "NB-001"
    well_conditions = [1]
    block_runs = ["block/run#1"]
    positions = ["R@1"]
    reps = [1]
    msmethods = ["dry"]
    filenames = mod.create_filenames(lc_number, conditions, nbcode, well_conditions, block_runs, positions, reps, msmethods)
    # Ensure special characters are preserved (or sanitized if code does so)
    assert any("NB-001_lib_block/run#1_R@1_rep1" in f for f in filenames)

def test_flattener_with_empty_nested_lists():
    mod = load_module()
    non_flat = [[], [[[]]]]
    flat = mod.flattener(non_flat)
    # Should return empty list or ignore empty items
    assert flat == []

def test_compare_wells_and_counts_with_exact_match(worklist_df):
    mod = load_module()
    nbcode, pathway, lc_number, wet_amounts, plates, num_to_run = mod.separate_plates(worklist_df)
    spacings, *_ = mod.spacing(worklist_df)
    name, plate_df = next(iter(plates.items()))
    wells = mod.process_plate(plate_df, name, wet_amounts)
    # Duplicate wells until match count exactly
    required_qcs = sum(spacings[0])  # total QC count expected
    wells_matched = wells * math.ceil(required_qcs / len([w for w in wells if w[0] == 1]))
    # Should not raise if counts match exactly
    mod.compare_wells_and_counts(wells_matched, mod.condition_dict(worklist_df), spacings, wet_amounts=None)
