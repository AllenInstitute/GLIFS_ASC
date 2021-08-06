import math

def count_params_glif(in_size, hid_size, out_size, num_asc, learnparams=True):
    if not learnparams:
        return count_params_rnn(in_size, hid_size, out_size)
    return (hid_size ** 2) + (hid_size * (in_size + out_size + (3 * num_asc) + 2))

def count_params_rnn(in_size, hid_size, out_size):
    return hid_size * (hid_size + in_size + 1)

def hid_size_glif(num_params, in_size, out_size, num_asc, learnparams):
    if not learnparams:
        return hid_size_rnn(num_params, in_size, out_size)
    a = 1
    b = in_size + out_size + (3 * num_asc) + 2
    c = -1 * num_params

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def hid_size_rnn(num_params, in_size, out_size):
    a = 1
    b = in_size + 1
    c = -1 * num_params

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))