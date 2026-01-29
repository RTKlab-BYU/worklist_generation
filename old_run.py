from metadata_capture.database_utils.upload_to_sqlite import upload_to_sqlite
from metadata_capture.excel_utils.excel_file_generator import BasicFileGenerator, CustomFileGenerator, AdvancedFileGenerator, \
    ExcelFileGenerator
from metadata_capture.excel_utils.excel_file_parser import ExcelFileParser, BasicFileParser, AdvancedFileParser, CustomFileParser
from metadata_capture.excel_utils.excel_utils import open_excel_sheet
from metadata_capture.project_dataclasses.project_metadata import ProjectMetadata
from metadata_capture.project_dataclasses.project_outline import ProjectOutline
from metadata_capture.cli_utils.cli_prompts import get_project_outline, get_project_type, ask_if_done_editing
from metadata_capture.project_dataclasses.project_type import ProjectType
from metadata_capture.vars import *
from metadata_capture.excel_utils.fill_conditions import fill_conditions_in_worklist   # ← add this import at top if not already present
import platform
import subprocess
import os



def main():

    # 1. Collect the Basic Project Info
    project_outline: ProjectOutline = get_project_outline()

    # 2. Get the Project Type (Basic, Advanced, or Custom)
    # project_type: ProjectType = get_project_type()
    # gen: ExcelFileGenerator
    # output: str
    project_type = ProjectType.ADVANCED

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
    print("opening metadata template...")
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

    project_id = upload_to_sqlite(project_metadata)

    print("\nGenerating filled worklist template with conditions...")

    try:
        # Adjust paths/filenames as needed — assuming project.db is at repo root
        filled_worklist_path = fill_conditions_in_worklist(
            project_id=project_id,
            db_path="project.db",
            template_path=str(TEMPLATES / "worklist_template_blank.xlsx"),
            output_dir="excel_utils/outputs",
            output_filename=f"worklist_conditions_project_{project_id}.xlsx"
        )

        print(f"Worklist saved to: {filled_worklist_path}")

        # Auto-open the filled file (cross-platform)
        if platform.system() == "Windows":
            os.startfile(filled_worklist_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", filled_worklist_path])
        else:  # Linux, etc.
            try:
                subprocess.call(["xdg-open", filled_worklist_path])
            except FileNotFoundError:
                print("Could not auto-open file (xdg-open not found). Please open it manually.")

        print("Worklist template opened for editing.")
        
        # if ask_if_done_editing(prompt="Are you finished editing the worklist?"):
        #     print("Worklist editing complete.")
        # else:
        #     print("Continuing without waiting for worklist confirmation...")

    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        print("→ Make sure 'project.db' and 'worklist_template_blank.xlsx' exist.")
    except Exception as e:
        print(f"Error while preparing/opening worklist: {e}")


if __name__ == '__main__':
    main()
