import sys
sys.path.append('..')

from template.gomoku import enjoy


def main():
    try:
        enjoy(int(sys.argv[1]))
    except Exception:
        print('Usage:')
        print('\tcd ./experiments')
        print('\tpython ./enjoy_kaithy board_size')


if __name__ == '__main__':
    main()
