import sys
import json
import numpy as np


def read_input() -> str:
    lines = sys.stdin.readlines()
    # Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])


def main():
    empty = '0'

    my_input = read_input()
    num_empty_square = my_input.count(empty)
    empty_move_idx = np.random.randint(num_empty_square)

    empty_square_count = -1
    for square_idx in range(0, len(my_input)):
        empty_square_count += (my_input[square_idx] == empty)
        print(empty_square_count)
        if empty_square_count == empty_move_idx:
            my_output = square_idx
            break

    print(my_output)


if __name__ == "__main__":
    main()
