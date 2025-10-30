from excel_utils.excel_utils import open_excel_sheet
from vars import BASE_DIR


def test_open_excel():
    path = BASE_DIR / "excel_utils" / "templates" / "TEST_EXCEL.xlsm"
    open_excel_sheet(path)
