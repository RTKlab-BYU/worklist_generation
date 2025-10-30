from project_classes.project_outline import ProjectOutline


def get_project_outline() -> ProjectOutline:
    name = input("What is your project name: ")
    description = input("What is your project description: ")
    while True:
        try:
            number_of_groups = int(input("How many groups does your project have: "))
            if not number_of_groups > 0:
                raise ValueError
            break
        except ValueError:
            print("Please insert a positive integer for number of groups.")

    groups = [input(f"Group {i} name: ") for i in range(number_of_groups)]

    return ProjectOutline(name=name, description=description, number_of_groups=number_of_groups, groups=groups)
