import os
import platform
import subprocess


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
