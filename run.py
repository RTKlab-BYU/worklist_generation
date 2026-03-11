#!/usr/bin/env python3

import sys
import os
import platform
import subprocess
import argparse
from pathlib import Path

# Stage 1 imports
from metadata_capture.cli_utils.cli_prompts import get_project_outline
from metadata_capture.project_dataclasses.project_type import ProjectType
from metadata_capture.excel_utils.excel_file_generator import AdvancedFileGenerator
from metadata_capture.excel_utils.excel_utils import open_excel_sheet
from metadata_capture.vars import TEMPLATES

# Stage 2 imports
from metadata_capture.excel_utils.excel_file_parser import AdvancedFileParser
from metadata_capture.database_utils.upload_to_sqlite import upload_to_sqlite
from metadata_capture.excel_utils.fill_conditions import fill_conditions_in_worklist

# Stage 3 imports
import worklist_classes.main as worklist_main


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run.py",
        usage="""
python run.py -s 1
python run.py -s 2 -m <metadata_excel_path>
python run.py -s 3 -w <worklist_excel_path> -o <output_dir>
""",
        description="LC/MS Worklist Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-s", "--stage",
        required=True,
        choices=["1", "2", "3"],
        help="Pipeline stage"
    )

    parser.add_argument(
        "-m", "--metadata",
        type=Path,
        help="Metadata Excel file (stage 2)"
    )

    parser.add_argument(
        "-w", "--worklist",
        type=Path,
        help="Worklist Excel file (stage 3)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (stage 3)"
    )

    return parser.parse_args()


def validate_args(args):
    if args.stage == "2" and not args.metadata:
        raise SystemExit("Stage 2 requires -m / --metadata")

    if args.stage == "3":
        if not args.worklist or not args.output:
            raise SystemExit("Stage 3 requires -w and -o")


# Helper Functions
def auto_open_file(path: Path):
    """Open file cross-platform."""
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        print(f"Could not auto-open file: {path}")


def usage():
    print("""
USAGE:
  python run.py -s 1
      → Generate metadata Excel

  python run.py -s 2 -m <metadata_excel_path>
      → Generate filled worklist Excel

  python run.py -s 3 -w <worklist_excel_path> -o <output_dir>
      → Generate LC + MS worklists
""")
    sys.exit(1)


# Stage 1
def stage_1_generate_metadata():
    """
    Collect project info and generate metadata Excel.
    """
    print("\n=== STAGE 1: Generate Metadata Excel ===\n")

    project_outline = get_project_outline()
    project_type = ProjectType.ADVANCED

    source_file = TEMPLATES / "metadataTemplate8.xlsm"
    generator = AdvancedFileGenerator(
        source_file=source_file,
        project_outline=project_outline
    )

    output_path = Path(generator.generate_file())

    print(f"\nMetadata Excel generated:\n  {output_path}")
    auto_open_file(output_path)

    print("\nNext step:")
    print(f"  python run.py -s 2 -m {output_path}\n")

    sys.exit(0)


# Stage 2
def stage_2_generate_worklist(metadata_excel_path: Path):
    """
    Parse metadata Excel, upload to DB, generate filled worklist Excel.
    """
    print("\n=== STAGE 2: Generate Filled Worklist ===\n")

    if not metadata_excel_path.exists():
        print(f"File not found: {metadata_excel_path}")
        sys.exit(1)

    parser = AdvancedFileParser(
        source_file=str(metadata_excel_path),
        project_outline=None  # parser reconstructs needed info internally
    )

    project_metadata = parser.parse_file()

    
    project_id = upload_to_sqlite(project_metadata)

    print(f"Project uploaded (ID: {project_id})")

    output_path = fill_conditions_in_worklist(
        project_id=project_id,
        db_path="project.db",
        template_path=str(TEMPLATES / "worklist_template_blank.xlsx"),
        output_dir="output",
        output_filename=f"worklist_conditions_project_{project_id}.xlsx"
    )

    output_path = Path(output_path)
    print(f"\nFilled worklist created:\n  {output_path}")
    auto_open_file(output_path)

    print("\nNext step:")
    print(f"  python run.py -s 3 -w {output_path} -o <output_dir>\n")

    sys.exit(0)


# Stage 3
def stage_3_generate_lcms(worklist_excel_path: Path, output_dir: Path):
    """
    Generate LC and MS worklist files.
    """
    print("\n=== STAGE 3: Generate LC/MS Worklists ===\n")

    if not worklist_excel_path.exists():
        print(f"File not found: {worklist_excel_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    ms_pd, lc_pd, ms_filename, lc_filename = worklist_main.main(
        str(worklist_excel_path)
    )

    ms_path = output_dir / ms_filename
    lc_path = output_dir / lc_filename

    ms_pd.to_csv(ms_path, index=False, header=False, encoding="utf-8-sig")
    lc_pd.to_csv(lc_path, index=False, header=False, encoding="utf-8-sig")

    print("\nWorklists generated:")
    print(f"  MS → {ms_path}")
    print(f"  LC → {lc_path}")

    sys.exit(0)


# Main
def main():
    print("hi")
    args = parse_args()
    validate_args(args)

    if args.stage == "1":
        stage_1_generate_metadata()

    elif args.stage == "2":
        if not args.metadata:
            print("Stage 2 requires --metadata")
            return
        stage_2_generate_worklist(args.metadata)

    elif args.stage == "3":
        if not args.worklist or not args.output:
            print("Stage 3 requires --worklist and --output")
            return
        stage_3_generate_lcms(args.worklist, args.output)
    else:
        usage()


if __name__ == "__main__":
    main()
