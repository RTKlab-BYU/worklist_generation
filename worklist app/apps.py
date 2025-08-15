import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import pyWorklist  # Your existing script
import sys
import shutil  # for copying template files

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller bundle"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  # PyInstaller sets this when bundled
    else:
        base_path = os.path.abspath(".")
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
        # Call your existing processing function
        output1, output2 = pyWorklist.process(file_path)

        save_dir = filedialog.askdirectory(title="Select folder to save results")
        if not save_dir:
            return

        output1_path = os.path.join(save_dir, "output1.csv")
        output2_path = os.path.join(save_dir, "output2.csv")

        output1.to_csv(output1_path, index=False)
        output2.to_csv(output2_path, index=False)

        messagebox.showinfo("Worklist Generator", "Files saved successfully!")
    except Exception as e:
        messagebox.showerror("Worklist Generator", str(e))

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
