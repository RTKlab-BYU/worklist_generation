from excel_utils.excel_file_generator import BasicFileGenerator, CustomFileGenerator, AdvancedFileGenerator, \
    ExcelFileGenerator
from excel_utils.excel_file_parser import ExcelFileParser, BasicFileParser, AdvancedFileParser, CustomFileParser
from excel_utils.excel_utils import open_excel_sheet
from project_classes.project_metadata import ProjectMetadata
from project_classes.project_outline import ProjectOutline
from cli_utils.cli_prompts import get_project_outline, get_project_type, ask_if_done_editing
from project_classes.project_type import ProjectType
from vars import *


def main():

    # Collect the Basic Project Info
    project_outline: ProjectOutline = get_project_outline()

    # Get the Project Type (Basic, Advanced, or Custom)
    project_type: ProjectType = get_project_type()
    gen: ExcelFileGenerator
    parser: ExcelFileParser
    output: str

    # Depending on Project Type, Use the correct Class
    if project_type == ProjectType.BASIC:
        source_file = TEMPLATES / "basic_project_template_1.xlsx"
        gen = BasicFileGenerator(source_file=source_file, project_outline=project_outline)
        output = gen.generate_file()
        # parser = BasicFileParser(source_file=output, project_outline=project_outline)

    elif project_type == ProjectType.CUSTOM:
        source_file = TEMPLATES / "CUSTOM_FILE_NOT_CREATED_YET.xlsx"
        gen = CustomFileGenerator(source_file=source_file, project_outline=project_outline)
        output = gen.generate_file()
        # parser = CustomFileParser(source_file=output, project_outline=project_outline)

    else:
        source_file = TEMPLATES / "metadataTemplate8.xlsm"
        gen = AdvancedFileGenerator(source_file=source_file, project_outline=project_outline)
        output = gen.generate_file()
        # parser = AdvancedFileParser(source_file=output, project_outline=project_outline)

    # Open the file
    open_excel_sheet(output)

    if ask_if_done_editing():
        print("NOT IMPLEMENTED YET; Quiting program....")
        quit()
    else:
        print("No Response. Quitting program...")
        quit()

    project_metadata: ProjectMetadata = parser.parse_file()


if __name__ == '__main__':
    main()
