from abc import ABC, abstractmethod

from project_classes.project_metadata import ProjectMetadata
from project_classes.project_outline import ProjectOutline


class ExcelFileParser(ABC):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        self.source_file = source_file
        self.project_outline = project_outline

    @abstractmethod
    def parse_file(self) -> ProjectMetadata:
        pass



class BasicFileParser(ExcelFileParser):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        super().__init__(source_file, project_outline)
        raise NotImplementedError

    def parse_file(self) -> ProjectMetadata:
        pass



class AdvancedFileParser(ExcelFileParser):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        super().__init__(source_file, project_outline)
        raise NotImplementedError

    def parse_file(self) -> ProjectMetadata:
        pass



class CustomFileParser(ExcelFileParser):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        super().__init__(source_file, project_outline)
        raise NotImplementedError

    def parse_file(self) -> ProjectMetadata:
        pass

