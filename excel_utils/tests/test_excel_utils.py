from excel_utils.excel_file_generator import AdvancedFileGenerator
from excel_utils.excel_utils import open_excel_sheet, copy_template_with_datetime
from project_classes.project_outline import ProjectOutline
from vars import BASE_DIR


def test_open_excel():
    path = BASE_DIR / "excel_utils" / "templates" / "TEST_EXCEL.xlsm"
    open_excel_sheet(path)


def test_copy_excel():
    copy_template_with_datetime(filename="basic_project_template_1.xlsx", new_name="project_name")



def test_advanced_excel():
    project_outline: ProjectOutline = ProjectOutline(name="TEST_PROJECT_1",
                                                     description="DESCRIPTION",
                                                     number_of_groups=3,
                                                     groups=["A", "B", "C"],
                                                     )

    gen = AdvancedFileGenerator("metadataTemplate8.xlsm", project_outline)

    file = gen.generate_file()

    open_excel_sheet(file)

