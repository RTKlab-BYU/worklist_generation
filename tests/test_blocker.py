import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worklist_classes.blocker import Blocker


def cond_row(label, before=0, after=0, between=0, method="MS_METHOD"):
    # Matches the shape created by excel_parser.condition_dict: indices 0..12 are used.
    return [label, "", "", "", method, "", "", "", "", "", before, after, between]


def make_blocker(
    all_conditions=None,
    conditions1=None,
    conditions2=None,
    wet_amounts=None,
    plates=None,
    num_to_run=None,
    lc_number=1,
    lib_placement="Before",
    sysvalid_interval=5,
    tb_location="R5",
    cond_range1="ALL",
    cond_range2="",
    lib_same="YES",
    even="YES",
    qc_frequency=2,
):
    all_conditions = all_conditions or {1: cond_row("Sample")}
    conditions = [all_conditions]
    if conditions1 is not None:
        conditions.append(conditions1)
    if conditions2 is not None:
        conditions.append(conditions2)

    parser_output = [
        pd.DataFrame(),
        pd.DataFrame(),
        lc_number,
        conditions,
        wet_amounts or {1: 1},
        plates or {},
        num_to_run or {1: "all"},
        lib_placement,
        sysvalid_interval,
        tb_location,
        cond_range1,
        cond_range2,
        lib_same,
        even,
        qc_frequency,
    ]
    return Blocker(parser_output)


def make_plate(values, rows, cols):
    return pd.DataFrame(values, index=rows, columns=cols)


def test_parse_range_handles_blank_and_valid_inputs():
    blocker = make_blocker()

    assert blocker.parse_range(None) == [0, 0]
    assert blocker.parse_range("") == [0, 0]
    assert blocker.parse_range("2-7") == [2, 7]


def test_parse_range_raises_on_invalid_value():
    blocker = make_blocker()

    with pytest.raises(ValueError, match="Invalid condition range"):
        blocker.parse_range("2 to 7")


def test_check_for_trueblank_adds_default_when_missing():
    blocker = make_blocker()
    conditions = {1: cond_row("Sample")}

    updated, tb_num, found = blocker.check_for_trueblank(conditions)

    assert found is False
    assert tb_num == 100
    assert updated[100][0] == "TrueBlank"


def test_process_plate_repeats_by_wet_amount_and_filters_by_range():
    blocker = make_blocker()
    plate = pd.DataFrame([[1, 2], [None, 1]], index=["A", "B"], columns=[1, 2])
    wet_amounts = {1: 2, 2: 1}

    all_wells = blocker.process_plate(plate, "R_plate1", wet_amounts)
    only_cond_1 = blocker.process_plate(plate, "R_plate1", wet_amounts, "1-1")

    assert len(all_wells) == 5
    assert len(only_cond_1) == 4
    assert all(w[0] == 1 for w in only_cond_1)
    assert {w[1] for w in all_wells} == {"RA1", "RA2", "RB2"}


def test_process_plate_raises_on_non_integer_column_name():
    blocker = make_blocker()
    plate = pd.DataFrame([[1]], index=["A"], columns=["A"])

    with pytest.raises(ValueError, match="must be a unique integer"):
        blocker.process_plate(plate, "R_plate1", {1: 1})


def test_process_plate_raises_on_non_whole_number_condition_id():
    blocker = make_blocker()
    plate = pd.DataFrame([["1.5"]], index=["A"], columns=[1])

    with pytest.raises(ValueError, match="Condition IDs must be whole numbers"):
        blocker.process_plate(plate, "R_plate1", {1: 1})


def test_process_plate_raises_when_condition_not_in_wet_amounts():
    blocker = make_blocker()
    plate = pd.DataFrame([[2]], index=["A"], columns=[1])

    with pytest.raises(KeyError, match="not labeled in the conditions legend"):
        blocker.process_plate(plate, "R_plate1", {1: 1})


def test_process_plate_raises_on_invalid_condition_range_string():
    blocker = make_blocker()
    plate = pd.DataFrame([[1]], index=["A"], columns=[1])

    with pytest.raises(ValueError, match="Invalid condition range"):
        blocker.process_plate(plate, "R_plate1", {1: 1}, "not-a-range")


def test_compare_wells_and_counts_raises_when_qc_is_insufficient():
    conditions = {1: cond_row("QC", before=1, after=1, between=0)}
    blocker = make_blocker(all_conditions=conditions, wet_amounts={1: 1})

    with pytest.raises(ValueError, match="Not enough QC wells"):
        blocker.compare_wells_and_counts([[1, "RA1"]], conditions, {1: 1})


def test_compare_wells_and_counts_raises_when_wet_amounts_missing():
    conditions = {1: cond_row("Sample")}
    blocker = make_blocker(all_conditions=conditions, wet_amounts={1: 1})

    with pytest.raises(ValueError, match="Samples/well must be provided"):
        blocker.compare_wells_and_counts([[1, "RA1"]], conditions, None)


def test_compare_wells_and_counts_requires_trueblank_presence_if_defined():
    conditions = {1: cond_row("TrueBlank")}
    blocker = make_blocker(all_conditions=conditions, wet_amounts={1: 1})

    with pytest.raises(ValueError, match="No TrueBlank wells found"):
        blocker.compare_wells_and_counts([], conditions, {1: 1})


def test_compare_wells_and_counts_requires_library_presence_if_defined():
    conditions = {1: cond_row("Library")}
    blocker = make_blocker(all_conditions=conditions, wet_amounts={1: 1})

    with pytest.raises(ValueError, match="No Library wells found"):
        blocker.compare_wells_and_counts([], conditions, {1: 1})


def test_compare_wells_and_counts_requires_systemvalidation_presence_if_defined():
    conditions = {1: cond_row("SystemValidation")}
    blocker = make_blocker(all_conditions=conditions, wet_amounts={1: 1})

    with pytest.raises(ValueError, match="No SystemValidation wells found"):
        blocker.compare_wells_and_counts([], conditions, {1: 1})


def test_column_sorter_splits_non_samples_and_applies_num_to_run():
    random.seed(42)
    all_conditions = {
        1: cond_row("Sample"),
        2: cond_row("Sample"),
        3: cond_row("QC", before=1),
        4: cond_row("Library"),
        5: cond_row("TrueBlank"),
        6: cond_row("SystemValidation"),
    }
    blocker = make_blocker(all_conditions=all_conditions, lc_number=2)

    wells = [
        [1, "RA1"], [1, "RA2"], [1, "RB1"], [1, "RB2"],
        [2, "RC1"], [2, "RC2"],
        [3, "RD1"], [3, "RD2"],
        [4, "RE1"], [4, "RE2"],
        [6, "RF1"], [6, "RF2"],
    ]

    out = blocker.column_sorter(
        wells_list=wells,
        conditions=all_conditions,
        num_to_run={1: 2, 2: "all"},
        lc_number=2,
        lib_placement="Before",
        cond_range1="ALL",
        found_TB=False,
        two_xp_TB=5,
    )

    nonsample_before, _, _, column1, column2, sysvalid_list, _ = out

    sample_wells = column1 + column2
    sample_ids = [w[0] for w in sample_wells]

    assert sample_ids.count(1) == 2
    assert sample_ids.count(2) == 2
    assert len(sysvalid_list) == 2
    print(nonsample_before)  # For debugging
    assert any(group == [[5, "R5"]] * 2 for group in nonsample_before)


def test_blocker_builds_blocks_for_one_column():
    random.seed(1)
    conditions = {1: cond_row("Sample"), 2: cond_row("Sample")}
    blocker = make_blocker(all_conditions=conditions, even="YES")

    column1 = [[1, "RA1"], [1, "RA2"], [2, "RB1"], [2, "RB2"]]
    both_blocks, num_blocks = blocker.blocker(conditions, "YES", column1)

    assert num_blocks == 2
    assert len(both_blocks) == 1
    assert all(len(block) == 2 for block in both_blocks[0])


def test_nonsample_blocker_one_column_splits_evenly():
    conditions = {3: cond_row("QC"), 4: cond_row("Blank")}
    blocker = make_blocker(all_conditions=conditions, even="YES")
    nonsample_other = [[3, "RA1"], [3, "RA2"], [4, "RB1"], [4, "RB2"]]

    blocks = blocker.nonsample_blocker(1, nonsample_other, num_of_blocks=2, conditions=conditions, total_wells=100)

    assert len(blocks) == 2
    assert all(len(b) == 2 for b in blocks)


def test_nonsample_blocker_two_column_singleton_can_trigger_divide_by_zero():
    conditions = {3: cond_row("QC")}
    blocker = make_blocker(all_conditions=conditions, even="YES")
    nonsample_other = [[3, "RA1"]]

    with pytest.raises(ZeroDivisionError):
        blocker.nonsample_blocker(2, nonsample_other, num_of_blocks=1, conditions=conditions, total_wells=100)


def test_zipper_interleaves_two_columns():
    blocker = make_blocker()
    both_blocks = [
        [[[1, "RA1"], [2, "RA2"]]],
        [[[1, "RB1"], [2, "RB2"]]],
    ]

    zipped = blocker.zipper(both_blocks)

    assert zipped == [[[1, "RA1"], [1, "RB1"], [2, "RA2"], [2, "RB2"]]]


def test_two_xp_zipper_requires_trueblank():
    blocker = make_blocker()

    with pytest.raises(ValueError, match="TrueBlank well must be specified"):
        blocker.two_xp_zipper([[[1, "RA1"]]], [[[2, "RB1"]]], None, {}, [[100, "R5"]])


def test_rep_tracker_counts_replicates_and_raises_for_unknown_condition():
    blocker = make_blocker()
    flattened = [[1, "RA1", "blo1"], [1, "RA2", "blo1"], ["1", "RA3", "blo2"], [1]]

    reps = blocker.rep_tracker(flattened, {1: cond_row("Sample")})
    assert reps == [1, 2, 3, None]

    with pytest.raises(KeyError, match="not found in conditions"):
        blocker.rep_tracker([[9, "RA9", "blo1"]], {1: cond_row("Sample")})


def test_block_end_to_end_single_experiment_without_excel_parser():
    random.seed(7)

    all_conditions = {
        1: cond_row("Sample", method="METHOD_A"),
        2: cond_row("SystemValidation", method="METHOD_SV"),
    }
    plate = pd.DataFrame([[1], [1], [1], [2], [2]], index=["A", "B", "C", "D", "E"], columns=[1])

    blocker = make_blocker(
        all_conditions=all_conditions,
        wet_amounts={1: 1, 2: 1},
        plates={"R_plate1": plate},
        num_to_run={1: "all", 2: "all"},
        lc_number=1,
        cond_range1="ALL",
        lib_same="YES",
        even="YES",
        qc_frequency=2,
    )

    blocked = blocker.block()

    well_conditions, block_runs, positions, reps, msmethods, lcmethods, conditions = blocked

    assert len(well_conditions) == len(block_runs) == len(positions) == len(reps) == len(msmethods) == len(lcmethods)
    assert set(well_conditions).issubset({1, 2, 100})
    assert "METHOD_A" in set(msmethods)
    assert 100 in conditions


def test_block_end_to_end_places_qc_pre_between_post_and_includes_lib_and_sysvalid():
    random.seed(11)
    all_conditions = {
        1: cond_row("Sample", method="METHOD_SAMPLE"),
        2: cond_row("QC", before=2, after=1, between=1, method="METHOD_QC"),
        3: cond_row("Library", method="METHOD_LIB"),
        4: cond_row("SystemValidation", method="METHOD_SV"),
        5: cond_row("TrueBlank", method="METHOD_TB"),
    }

    # 26 wells total (13x2): Sample x8, QC x7, Lib x2, SysValid x8, TrueBlank x1
    values = (
        [[1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 3],
         [3, 4], [4, 4], [4, 4], [4, 4], [5, 4]]
    )
    plate = make_plate(values, list("ABCDEFGHIJKLM"), [1, 2])

    blocker = make_blocker(
        all_conditions=all_conditions,
        wet_amounts={1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
        plates={"R_plate1": plate},
        num_to_run={1: "all", 2: "all", 3: "all", 4: "all", 5: "all"},
        lc_number=1,
        lib_placement="Before",
        sysvalid_interval=4,
        cond_range1="ALL",
        lib_same="YES",
        even="YES",
        qc_frequency=2,
    )

    well_conditions, block_runs, positions, reps, msmethods, lcmethods, _ = blocker.block()

    assert len(well_conditions) == len(block_runs) == len(positions) == len(reps) == len(msmethods) == len(lcmethods)
    assert any(c == 3 for c in well_conditions)  # Lib is present
    assert any(c == 4 and r == "SysQC" for c, r in zip(well_conditions, block_runs))  # SysValid inserted

    qc_tags = {r for c, r in zip(well_conditions, block_runs) if c == 2}
    assert "pre" in qc_tags
    assert "other" in qc_tags
    assert "post" in qc_tags


def test_block_end_to_end_lib_after_tags_library_as_post():
    random.seed(17)
    all_conditions = {
        1: cond_row("Sample"),
        2: cond_row("QC", before=1, after=1, between=0),
        3: cond_row("Library"),
        4: cond_row("SystemValidation"),
    }
    values = [[1, 1], [1, 1], [2, 2], [3, 3], [4, 4]]
    plate = make_plate(values, list("ABCDE"), [1, 2])

    blocker = make_blocker(
        all_conditions=all_conditions,
        wet_amounts={1: 1, 2: 1, 3: 1, 4: 1},
        plates={"R_plate1": plate},
        num_to_run={1: "all", 2: "all", 3: "all", 4: "all"},
        lc_number=1,
        lib_placement="After",
        sysvalid_interval=3,
        cond_range1="ALL",
        lib_same="YES",
        even="YES",
        qc_frequency=2,
    )

    well_conditions, block_runs, *_ = blocker.block()
    lib_tags = {r for c, r in zip(well_conditions, block_runs) if c == 3}

    assert lib_tags == {"post"}


def test_block_end_to_end_two_column_inserts_systemvalidation_in_pairs():
    random.seed(23)
    all_conditions = {
        1: cond_row("Sample"),
        4: cond_row("SystemValidation"),
    }
    values = [[1, 1], [1, 1], [1, 1], [1, 1], [4, 4], [4, 4]]
    plate = make_plate(values, list("ABCDEF"), [1, 2])

    blocker = make_blocker(
        all_conditions=all_conditions,
        wet_amounts={1: 1, 4: 1},
        plates={"R_plate1": plate},
        num_to_run={1: "all", 4: "all"},
        lc_number=2,
        lib_placement="Before",
        sysvalid_interval=2,
        cond_range1="ALL",
        lib_same="YES",
        even="YES",
        qc_frequency=2,
    )

    well_conditions, block_runs, *_ = blocker.block()
    sysqc_conditions = [c for c, r in zip(well_conditions, block_runs) if r == "SysQC"]

    assert len(sysqc_conditions) >= 2
    assert len(sysqc_conditions) % 2 == 0
    assert set(sysqc_conditions) == {4}


def test_block_raises_for_invalid_configuration():
    blocker = make_blocker(
        all_conditions={1: cond_row("Sample")},
        wet_amounts={1: 1},
        plates={"R_plate1": pd.DataFrame([[1]], index=["A"], columns=[1])},
        num_to_run={1: "all"},
        lc_number=1,
        cond_range1="1-1",
        cond_range2="",
        lib_same="YES",
    )

    with pytest.raises(ValueError, match="Experiment conditions cannot be run"):
        blocker.block()


def test_block_without_systemvalidation_condition_hits_unpacking_error():
    # Represents a plausible parser output where no SystemValidation row was defined.
    blocker = make_blocker(
        all_conditions={1: cond_row("Sample")},
        wet_amounts={1: 1},
        plates={"R_plate1": pd.DataFrame([[1], [1]], index=["A", "B"], columns=[1])},
        num_to_run={1: "all"},
        lc_number=1,
        cond_range1="ALL",
        lib_same="YES",
        even="YES",
        qc_frequency=2,
    )

    with pytest.raises(ValueError, match="not enough values to unpack"):
        blocker.block()
