from copy import deepcopy

from metadata_capture.excel_utils import AdvancedFileGenerator
from metadata_capture.excel_utils.excel_file_parser import AdvancedFileParser
from metadata_capture.excel_utils.excel_utils import open_excel_sheet, copy_template_with_datetime
from metadata_capture.project_dataclasses.project_outline import ProjectOutline
from metadata_capture.vars import BASE_DIR, TEMPLATES


def test_open_excel():
    path = BASE_DIR / "excel_utils" / "templates" / "TEST_EXCEL.xlsm"
    open_excel_sheet(path)


def test_copy_excel():
    copy_template_with_datetime(filename="basic_project_template_1.xlsx", new_name="project_name")


"""
/Users/calebcoons/Projects/cli-metadata-generation/.venv/bin/pytest -v excel_utils/tests/test_excel_utils.py::test_advanced_excel
"""


def test_advanced_excel():
    project_outline: ProjectOutline = ProjectOutline(name="TEST_PROJECT_1",
                                                     description="DESCRIPTION",
                                                     number_of_groups=3,
                                                     groups=["A", "B", "C"],
                                                     )

    gen = AdvancedFileGenerator("metadataTemplate8.xlsm", project_outline)

    file = gen.generate_file()

    open_excel_sheet(file)


def normalize(d):
    if isinstance(d, dict):
        return {k.strip(): normalize(v) for k, v in d.items()}
    elif isinstance(d, list):
        return sorted(normalize(x) for x in d)
    return d


def test_advanced_parser_expected_vs_result():
    project_outline = ProjectOutline(
        name="a",
        description="a",
        number_of_groups=3,
        groups=["a", "b", "c"],
    )

    file = TEMPLATES / "TEST_EDITED_FILE_1.xlsm"
    parser = AdvancedFileParser(source_file=file, project_outline=project_outline)
    metadata = parser.parse_file()

    # Normalize results (remove defaultdict / strip spaces)
    result_metadata = normalize(deepcopy(metadata.metadata))
    result_indvars = normalize(deepcopy(metadata.independent_variables))

    category = next(iter(result_metadata["a"].keys()))

    expected_metadata = {
        "a": {category: {"Species": "a", "Sex": "l", "Age": "j", "Gene": "k", "Tissue": "j"}},
        "b": {category: {"Species": "b", "Sex": "l", "Age": "j", "Gene": "k", "Tissue": "k"}},
        "c": {category: {"Species": "c", "Sex": "l", "Age": "j", "Gene": "k", "Tissue": "y"}},
    }

    expected_indvars = {
        category: {
            "Species": ["a", "b", "c"],
            "Tissue": ["j", "k", "y"],
        }
    }

    expected_metadata = normalize(expected_metadata)
    expected_indvars = normalize(expected_indvars)

    # *** These two assertions will produce a great diff ***
    assert result_metadata == expected_metadata
    assert result_indvars == expected_indvars


def test_advanced_parser_output_print_results():
    project_outline = ProjectOutline(
        name="a",
        description="a",
        number_of_groups=3,
        groups=["a", "b", "c"],
    )

    file = TEMPLATES / "TEST_EDITED_FILE_1.xlsm"
    parser = AdvancedFileParser(source_file=file, project_outline=project_outline)
    metadata = parser.parse_file()

    print(metadata)