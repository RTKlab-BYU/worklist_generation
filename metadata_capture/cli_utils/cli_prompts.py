from metadata_capture.project_dataclasses.project_outline import ProjectOutline
from metadata_capture.project_dataclasses.project_type import ProjectType


def get_project_outline() -> ProjectOutline:
    name = input("What is your project name: ")
    description = input("What is your project description: ")
    while True:
        try:
            number_of_groups = int(input("How many conditions does your project have: "))
            if not number_of_groups > 0:
                raise ValueError
            break
        except ValueError:
            print("Please insert a positive integer for number of conditions.")

    groups = [input(f"Condition {i + 1} name: ") for i in range(number_of_groups)]

    return ProjectOutline(name=name, description=description, number_of_groups=number_of_groups, groups=groups)


def get_project_type() -> ProjectType:
    resp = input("Select Project Type: \n"
                 "(1) - Basic\n"
                 "(2) - Advanced\n"
                 "(3) - Custom\n")
    match resp:
        case "1":
            return ProjectType.BASIC
        case "2":
            return ProjectType.ADVANCED
        case "3":
            return ProjectType.CUSTOM
        case _:
            raise ValueError("Must Select 1, 2, or 3")


def ask_if_done_editing() -> bool:
    if input("Press 'ENTER' When You Have Finished Editing Your Excel AND saved the file (cmd/ctrl + 'S')") not in ["quit", "q", "quit()"]:
        return True
    return False
