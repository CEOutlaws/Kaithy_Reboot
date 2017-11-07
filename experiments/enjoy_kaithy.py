import sys
sys.path.append('..')

from template.gomoku import enjoy


def main():
    try:
        enjoy(
            board_size=int(sys.argv[1])
        )
    except Exception as e:
        print('Usage:')
        print('\tcd ./experiments')
        print('\tpython ./enjoy_kaithy board_size')


if __name__ == '__main__':
    main()
