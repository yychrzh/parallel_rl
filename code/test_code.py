import numpy as np
import time
import tensorflow as tf


def test_dict():
    a, b, c, d = 0, 1, 2, 3
    var1 = tf.placeholder(tf.float32, name='var1')
    var2 = tf.placeholder(tf.float32, name='var2')
    dict_1 = {var1: 0}
    dict_2 = {var2: 1}
    dictMerged = {}  # dict_1.copy()
    dictMerged.update(dict_1)
    dictMerged.update(dict_2)
    print(dictMerged)


def test_dict_items():
    dict_0 = {'1': [1], '2': [2, 3], '3': [3, 4, 5]}
    print(dict_0)
    for key, value in dict_0.items():
        dict_0[key] = np.array(value)
    print(dict_0)


def test_args():
    import argparse
    # parameter list
    param_list = {
        "task":                str('Pendulum-v0'),
        "model":               str('trpo'),
        "timesteps_per_batch": int(5000),
        "n_steps":             int(100000),
        "n_episodes":          int(10000),
        "max_pathlength":      int(200),
    }

    parser = argparse.ArgumentParser(description='parallel_rl')
    for key, value in param_list.items():
        parser.add_argument('--' + key, type=type(value), default=value)

    args = parser.parse_args()
    for key, _ in param_list.items():
        value = getattr(args, key)
        print(key + ': ', value)


def test_add_dict():
    path = {}
    path["act"] = [0, 1, 2]
    path["obs"] = [0, 3, 4]
    print(path)


if __name__ == '__main__':
    # test_dict()
    # test_dict_items()
    # test_args()
    test_add_dict()