from excel_parser import ExcelParser
from blocker import Blocker
from output import Output

def main(input_filename):
    parser = ExcelParser(input_filename)
    parsed_for_blocker, parsed_for_output = parser.parse()

    blocker = Blocker(parsed_for_blocker)
    blocked_for_output = blocker.block()

    output = Output(parsed_for_output, blocked_for_output)
    ms_pd, lc_pd, ms_filename, lc_filename = output.putout()
    return ms_filename, lc_filename

if __name__ == "__main__":
    main("worklist_template0821.xlsx")
