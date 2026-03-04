### The purpose of this script is to run pyWorklist.py through the command line

# input format to run program ex: python3 commandline.py -r <input_file> <save_dir(optional)> 
# input format to download template ex: python3 commandline.py -t <new_template_name(optional)>

# for testing copy: python3 commandline.py -r "C:\Users\ivint\OneDrive\Desktop\Kelly_Lab\.venv\worklist_git\worklist_template0821.xlsx" "C:\Users\ivint\OneDrive\Desktop\Kelly_Lab\.venv\worklist_git\output_worklist"


import pyWorklist
import os
import sys
import worklist_classes.main as main

def save_template_copy(template_path, new_file_path):
    """Save template to Downloads folder"""
    try:
        with open(template_path, "rb") as src_file:
            with open(new_file_path, "wb") as dest_file:
                dest_file.write(src_file.read())
        print(f"Template saved to {new_file_path}\n")
    except Exception as e:
        print(f"Failed to save template:\n{e}\n")

def create_template_copy(new_name=None):
    """Create a copy of the template with new name if provided"""
    #template_path = "worklist_git\\worklist_template_blank.xlsx"
    template_path = "worklist_template_blank.xlsx"

    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    if new_name:
        if not new_name.lower().endswith(".xlsx"):
                new_name += ".xlsx"
    else:
        new_name = "worklist_template_blank_copy.xlsx"
    new_file_path = os.path.join(downloads_folder, new_name)
    save_template_copy(template_path, new_file_path)

def generate_worklist(input_file, output_path=None):
    """Generates worklist and saves file to downloads folder or specified path"""
    try:
        # ms_pd, lc_pd, ms_filename, lc_filename = pyWorklist.process(input_file)
        ms_pd, lc_pd, ms_filename, lc_filename = main.main(input_file)

        if output_path is None:
                # downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
                # output_path = os.path.join(downloads_folder, "worklist_output.xlsx")
                output_path = os.path.join(os.path.expanduser("~"), "Downloads")
        # create file paths 
        output1_path = os.path.join(output_path, ms_filename)
        output2_path = os.path.join(output_path, lc_filename)

        # Save in Excel-friendly CSV format
        ms_pd.to_csv(output1_path, index=False, header=False, encoding="utf-8-sig", lineterminator="\r\n")
        lc_pd.to_csv(output2_path, index=False, header=False, encoding="utf-8-sig", lineterminator="\r\n")

        print(f"{ms_filename}, {lc_filename} saved successfully!")
    except Exception as e:
        print(f"Worklist failed:\n{e}")

    


if __name__ == "__main__":
    if sys.argv[1] == '-t':
        if len(sys.argv) > 2:
            new_name = sys.argv[2]
            create_template_copy(new_name)
        else:
            create_template_copy()

    elif sys.argv[1] == '-r':
        input_file = sys.argv[2]
        if len(sys.argv) > 3:
        # if sys.argv[3]:
            output_path = sys.argv[3]
            generate_worklist(input_file, output_path)
        else:
            generate_worklist(input_file)
