from project_classes.project_metadata import ProjectMetadata
from project_classes.project_outline import ProjectOutline
from cli_utils.cli_prompts import get_project_outline


def main():

    # Collect the Basic Project Info
    project_outline: ProjectOutline = get_project_outline()

    # Create the Metadata Input Sheet, Return the filepath
    metadata_excel_path: str = create_metadata_excel(project_outline)

    # Open the file
    open_excel_sheet(metadata_excel_path)


    if ask_if_done_editing():
        parse_result: MetadataParseResult = parse_metadata_excel(project_outline, metadata_excel_path)


    project_metadata: ProjectMetadata = parse_result.project_metadata


if __name__ == '__main__':
    main()
