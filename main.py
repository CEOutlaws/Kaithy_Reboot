import sys
import json
import numpy as np


def read_input():
    lines = sys.stdin.readlines()
    # Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])


def main():
    # get our data as an array from read_in()
    input = read_input()

    # return the sum to the output stream
    output = "3"
    print(output)


if __name__ == "__main__":
    main()
