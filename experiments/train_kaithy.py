import sys
sys.path.append('..')

from template.gomoku import train


def main():
    train(
        board_size=int(sys.argv[1]),
        max_timesteps=int(sys.argv[2])
    )
    try:
        train(
            board_size=int(sys.argv[1]),
            max_timesteps=int(sys.argv[2])
        )
    except Exception as e:
        print('Usage:')
        print('\tcd ./experiments')
        print('\tpython ./train_kaithy board_size max_time_step')


if __name__ == '__main__':
    main()
