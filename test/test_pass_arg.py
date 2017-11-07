import sys
sys.path.append('..')

import adversarial_gym as gym
import numpy as np
import tensorflow as tf


def my_func(arg_0, arg_1, arg_2, arg_3):
    print(arg_0, arg_1, arg_2, arg_3)


def main():
    '''
    AI Self-training program
    '''
    kwargs = {}
    kwargs['arg_0'] = 0
    kwargs['arg_1'] = 1
    kwargs['arg_2'] = 2
    kwargs['arg_3'] = 3

    my_func(*kwargs)
    my_func(**kwargs)

    kwargs = {}
    kwargs['arg_0'] = 0
    kwargs['arg_1'] = 1
    kwargs['arg_2'] = 2

    my_func(arg_3=-1, *kwargs)
    my_func(arg_3=3, **kwargs)

    kwargs = {}
    kwargs['arg_1'] = 1
    kwargs['arg_2'] = 2

    my_func(-2, arg_3=-1, *kwargs)
    my_func(-2, arg_3=3, **kwargs)


if __name__ == "__main__":
    main()
