import sys
sys.path.append('..')

from template.gomoku import enjoy


def main():
    try:
        enjoy(sys.argv[1])
    except Exception:
        print('Usage:')
        print('\tcd ./experients')
        print('\tpython ./enjoy_kaithy board_size')


if __name__ == '__main__':
    main()
