from abc import ABC, abstractmethod
from collections import defaultdict

import openpyxl

from project_classes.project_metadata import ProjectMetadata
from project_classes.project_outline import ProjectOutline


class ExcelFileParser(ABC):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        self.source_file = source_file
        self.project_outline = project_outline
        self.wb = openpyxl.load_workbook(self.source_file, read_only=True)
        self.ws = self.wb["DataEntry"]


    def parse_file(self) -> ProjectMetadata:
        return self._parse_file()

    @abstractmethod
    def _parse_file(self) -> ProjectMetadata:
        pass


class BasicFileParser(ExcelFileParser):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        super().__init__(source_file, project_outline)
        raise NotImplementedError

    def _parse_file(self) -> ProjectMetadata:
        pass



class AdvancedFileParser(ExcelFileParser):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        super().__init__(source_file, project_outline)
        raise NotImplementedError

    def _parse_file(self) -> ProjectMetadata:

        """
        Go Row By Row
        For each Row:
            Read Across
            Determine if it should be considered an independent variable or not
            add it to the dictionary according to it's group
        """
        cur_row = 9
        category = self.ws.cell(row=cur_row, column=1).value
        next_row_new_category = False

        while True:
            cat_col = self.ws.cell(row=cur_row, column=1).value
            if cat_col == "END_OF_FILE":
                break
            values: tuple[str, list[tuple[str, str]]] = self.read_row(cur_row)

            if values:
                pass



            cur_row += 1



        return ProjectMetadata(self.project_outline,
                               independent_variables=,
                               metadata=,
                               )

    def read_row(self, cur_row) -> tuple[str, list[tuple[str, str]]]:
        label = self.ws.cell(row=cur_row, column=2).value

        if not label:
            return "", []

        for index, group in enumerate(self.project_outline.groups):
            col = index + 3
            val = self.ws.cell(row=cur_row, column=col)







class CustomFileParser(ExcelFileParser):

    def __init__(self, source_file: str, project_outline: ProjectOutline):
        super().__init__(source_file, project_outline)
        raise NotImplementedError

    def _parse_file(self) -> ProjectMetadata:
        pass

