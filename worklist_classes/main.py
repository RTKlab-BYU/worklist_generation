from excel_parser import ExcelParser
from blocker import Blocker
from output import Output

def main(input_filename):
    parser = ExcelParser(input_filename)
    parsed_for_blocker, parsed_for_output = parser.parse()

    blocker = Blocker(parsed_for_blocker)
    blocked_for_output = blocker.block()

    output = Output(parsed_for_output, blocked_for_output)
    return output.putout()

if __name__ == "__main__":
    main()
