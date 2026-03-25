### The purpose of this script is to run worklist_classes/main.py through the command line

# input format to download template ex: python3 commandline.py -t <new_template_name(optional)>
# input format to run program ex: python3 commandline.py -r <input_file> <save_dir(optional)> 

# for testing copy: python commandline.py -r "C:\Users\ivint\OneDrive\Desktop\Kelly_Lab\.venv\worklist_git\worklist_template0821.xlsx" "C:\Users\ivint\OneDrive\Desktop\Kelly_Lab\.venv\worklist_git\output_worklist"


import worklist_classes.main as main
import os
import sys

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
        ms_pd, lc_pd, ms_filename, lc_filename = main.main(input_file)
        if output_path is None:
                downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
                output_path = os.path.join(downloads_folder, "worklist_output.xlsx")
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
        if sys.argv[3]:
            output_path = sys.argv[3]
            generate_worklist(input_file, output_path)
        else:
            generate_worklist(input_file)



        # new_template_path = os.path.join(downloads_folder, "worklist_template_blank.xlsx")

        # # get new file name from user
        # new_name = input("Enter a name for the template file (or hit enter to use default name): ")
        # if new_name:
        #     if not new_name.lower().endswith(".xlsx"):
        #         new_name += ".xlsx"
        #     new_template_path = os.path.join(downloads_folder, new_name)
        #     try:
        #         with open(template_path, "rb") as src_file:
        #             with open(new_template_path, "wb") as dest_file:
        #                 dest_file.write(src_file.read())
        #         print(f"Template saved to {new_template_path}\n")
        #     except Exception as e:
        #         print(f"Failed to save template:\n{e}\n")
        # else:
        #     try:
        #         new_name = "worklist_template_blank_copy.xlsx"
        #         new_template_path = os.path.join(downloads_folder, new_name)

        #         with open(template_path, "rb") as src_file:
        #             with open(new_template_path, "wb") as dest_file:
        #                 dest_file.write(src_file.read())
        #         print(f"Template saved to {new_template_path}\n")
        #     except Exception as e:
        #         print(f"Failed to save template:\n{e}\n")


    # check if user wants to run the generator
    # run = input("Do you want to run the worklist generator? (y/n): ")
    # if run.lower() == 'y':
    #     input_file = input("Enter the path to your worklist .xlsx file: ")
    #     input_file = input_file.strip('"') # copying the path from explorer adds quotes
    #     print(input_file)

    #     try:
    #         # Call the processing function from pyWorklist
    #         ms_pd, lc_pd, ms_filename, lc_filename = pyWorklist.process(input_file)

    #         # save_dir = input("Enter name of folder to save results: ")
    #         while True:
    #             save_dir = input("Enter name of folder to save results (or hit enter to save to current folder): ")

    #             if not save_dir:
    #                 save_dir = os.getcwd()
    #                 break
                
    #             if os.path.isdir(save_dir):
    #                 break

    #             print(f'Folder "{save_dir}" not found.')
    #             new_folder = input("Do you want to create it? (y/n): ")
    #             if new_folder.lower() == 'y':
    #                 try:
    #                     os.makedirs(save_dir)
    #                     print(f'Folder "{save_dir}" created.')
    #                     break
    #                 except Exception as e:
    #                     print(f"Failed to create folder:\n{e}")
                
    #         # create file paths 
    #         output1_path = os.path.join(save_dir, ms_filename)
    #         output2_path = os.path.join(save_dir, lc_filename)

    #         # Save in Excel-friendly CSV format
    #         ms_pd.to_csv(output1_path, index=False, header=False, encoding="utf-8-sig", lineterminator="\r\n")
    #         lc_pd.to_csv(output2_path, index=False, header=False, encoding="utf-8-sig", lineterminator="\r\n")

    #         print(f"{ms_filename}, {lc_filename} saved successfully!")
            
    #     except Exception as e:
    #         print(f"Worklist generation failed:\n{e}")

