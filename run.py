from database_utils.upload_to_sqlite import upload_to_sqlite
from excel_utils.excel_file_generator import BasicFileGenerator, CustomFileGenerator, AdvancedFileGenerator, \
    ExcelFileGenerator
from excel_utils.excel_file_parser import ExcelFileParser, BasicFileParser, AdvancedFileParser, CustomFileParser
from excel_utils.excel_utils import open_excel_sheet
from project_dataclasses.project_metadata import ProjectMetadata
from project_dataclasses.project_outline import ProjectOutline
from cli_utils.cli_prompts import get_project_outline, get_project_type, ask_if_done_editing
from project_dataclasses.project_type import ProjectType
from vars import *


def main():

    # 1. Collect the Basic Project Info
    project_outline: ProjectOutline = get_project_outline()

    # 2. Get the Project Type (Basic, Advanced, or Custom)
    project_type: ProjectType = get_project_type()
    gen: ExcelFileGenerator
    output: str

    # 3. Depending on Project Type, Use the correct generator and parser class
    if project_type == ProjectType.BASIC:
        raise NotImplementedError
        source_file = TEMPLATES / "basic_project_template_1.xlsx"
        gen = BasicFileGenerator(source_file=source_file, project_outline=project_outline)
        output = gen.generate_file()
        parser = BasicFileParser(source_file=output, project_outline=project_outline)

    elif project_type == ProjectType.CUSTOM:
        raise NotImplementedError
        source_file = TEMPLATES / "CUSTOM_FILE_NOT_CREATED_YET.xlsx"
        gen = CustomFileGenerator(source_file=source_file, project_outline=project_outline)
        output = gen.generate_file()
        parser = CustomFileParser(source_file=output, project_outline=project_outline)

    # Default to Advanced
    else:
        source_file = TEMPLATES / "metadataTemplate8.xlsm"
        gen = AdvancedFileGenerator(source_file=source_file, project_outline=project_outline)
        output = gen.generate_file()

    # 4. Open the file
    open_excel_sheet(output)

    # 5. Wait until they are done editing the Excel
    if ask_if_done_editing():
        pass
    else:
        print("No Response. Quitting program...")
        quit()

    # 6. Parse the Excel File, return a ProjectMetadata class
    project_metadata: ProjectMetadata
    parser: ExcelFileParser

    if project_type == ProjectType.BASIC:
        parser = BasicFileParser(source_file=output, project_outline=project_outline)

    elif project_type == ProjectType.CUSTOM:
        parser = CustomFileParser(source_file=output, project_outline=project_outline)

    else:
        parser = AdvancedFileParser(source_file=output, project_outline=project_outline)

    # parse_file() function called
    project_metadata = parser.parse_file()
    print(project_metadata)

    upload_to_sqlite(project_metadata)


if __name__ == '__main__':
    main()
