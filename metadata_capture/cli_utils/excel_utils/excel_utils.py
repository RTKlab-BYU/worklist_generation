import platform
import shutil
import subprocess
from datetime import datetime
import os
from typing import Optional
from vars import BASE_DIR


def open_excel_sheet(filepath: str):
    """
    Opens an .xlsm file in Excel (or the system's default spreadsheet program)
    across macOS, Windows, and Linux.
    """
    system = platform.system()

    if system == "Windows":
        os.startfile(filepath)

    elif system == "Darwin":  # macOS
        # Try to open specifically with Microsoft Excel if installed
        excel_path = "/Applications/Microsoft Excel.app"
        if os.path.exists(excel_path):
            subprocess.run(["open", "-a", "Microsoft Excel", filepath])
        else:
            subprocess.run(["open", filepath])  # fallback to default app

    elif system == "Linux":
        # Use xdg-open to open with the default spreadsheet app
        subprocess.run(["xdg-open", filepath])

    else:
        raise OSError(f"Unsupported operating system: {system}")


def copy_template_with_datetime(
    filename: str,
    new_name: Optional[str] = None
) -> str:
    """
    Copies a template Excel file (.xlsx or .xlsm) from the templates folder to the outputs folder,
    naming the new file with the current datetime and an optional custom name.

    Args:
        filename (str): Name of the template file in the 'templates' folder (e.g., 'template.xlsm').
        new_name (str, optional): Optional name to append after datetime.

    Returns:
        str: Full path to the new copied file in the 'outputs' folder.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
        ValueError: If the file is not a recognized Excel type.
    """
    templates_dir = BASE_DIR / "excel_utils" / "templates"
    outputs_dir = BASE_DIR / "excel_utils" / "outputs"

    # Ensure outputs directory exists
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve full path to template
    template_path = (templates_dir / filename).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Validate it's an Excel file
    if template_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(f"Unsupported file type: {template_path.suffix}. Only .xlsx and .xlsm are allowed.")

    # Get current datetime
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build new filename
    name_parts = [datetime_str]
    if new_name:
        # Sanitize new_name to avoid invalid filename chars
        safe_new_name = "".join(c if c.isalnum() or c in " _-." else "_" for c in new_name.strip())
        if safe_new_name:
            name_parts.append(safe_new_name)

    new_filename = "_".join(name_parts) + template_path.suffix
    new_file_path = outputs_dir / new_filename

    # Avoid overwrite (optional: add counter if needed)
    counter = 1
    original_path = new_file_path
    while new_file_path.exists():
        new_filename = f"{original_path.stem}_{counter}{original_path.suffix}"
        new_file_path = outputs_dir / new_filename
        counter += 1

    # Copy file preserving metadata
    shutil.copy2(template_path, new_file_path)

    return str(new_file_path)


"""
Example Usage
# Copy .xlsm
path1 = copy_template_with_datetime("report_template.xlsm", "Q3_Summary")
# → outputs/20251030_143022_Q3_Summary.xlsm

# Copy .xlsx
path2 = copy_template_with_datetime("data_input.xlsx")
# → outputs/20251030_143025.xlsx
"""
