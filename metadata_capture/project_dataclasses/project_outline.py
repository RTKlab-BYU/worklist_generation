from dataclasses import dataclass


@dataclass
class ProjectOutline:
    name: str
    description: str
    number_of_groups: int
    groups: list[str]

