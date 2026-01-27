import sys, os, traceback

logfile = os.path.expanduser("~/worklist_generator_log.txt")
sys.stderr = open(logfile, "a", buffering=1)  # line-buffered append
sys.stdout = sys.stderr


import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import cruft.pyWorklist as pyWorklist
import shutil  # for copying template files

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller bundle"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(base_path, relative_path)

# Path to the bundled template file
TEMPLATE_FILE = resource_path("data/worklist_template.xlsx")

def download_template():
    """Allow user to save the bundled worklist template to their machine"""
    try:
        save_path = filedialog.asksaveasfilename(
            title="Save Worklist Template",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="worklist_template.xlsx"
        )
        if save_path:
            shutil.copyfile(TEMPLATE_FILE, save_path)
            messagebox.showinfo("Worklist Generator", "Template downloaded successfully!")
    except Exception as e:
        messagebox.showerror("Worklist Generator", f"Failed to download template:\n{e}")

def run_worklist():
    """Run the main worklist process"""
    file_path = filedialog.askopenfilename(
        title="Select your worklist .xlsx file",
        filetypes=[("Excel files", "*.xlsx")]
    )
    if not file_path:
        return

    try:
        # Call the processing function from pyWorklist
        ms_pd, lc_pd, ms_filename, lc_filename = pyWorklist.process(file_path)

        save_dir = filedialog.askdirectory(title="Select folder to save results")
        if not save_dir:
            return

        output1_path = os.path.join(save_dir, ms_filename)
        output2_path = os.path.join(save_dir, lc_filename)

        # Save in Excel-friendly CSV format
        ms_pd.to_csv(output1_path, index=False, header=False, encoding="utf-8-sig", lineterminator="\r\n")
        lc_pd.to_csv(output2_path, index=False, header=False, encoding="utf-8-sig", lineterminator="\r\n")

        messagebox.showinfo("Worklist Generator", f"{ms_filename}, {lc_filename} saved successfully!")

    except Exception as e:
        # Write full traceback to logfile
        traceback.print_exc(file=sys.stderr)

        # Also show a popup with the simple message
        messagebox.showerror(
            "Worklist Generator",
            f"An error occurred:\n{e}\n\nSee {logfile} for details."
        )


# --- UI Setup ---
root = tk.Tk()
root.title("Worklist Generator")
root.geometry("300x250")

label = tk.Label(root, text="Worklist Generator", font=("Arial", 14))
label.pack(pady=10)

# Button to download template
template_button = tk.Button(root, text="Download Template", command=download_template, width=20, height=2)
template_button.pack(pady=10)

# Button to run worklist
worklist_button = tk.Button(root, text="Run Worklist", command=run_worklist, width=20, height=2)
worklist_button.pack(pady=10)

root.mainloop()
