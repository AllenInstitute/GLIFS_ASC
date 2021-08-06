import math

def count_params_glif(in_size, hid_size, out_size, num_asc, learnparams=True):
    if not learnparams:
        return count_params_rnn(in_size, hid_size, out_size)
    return (hid_size ** 2) + (hid_size * (in_size + out_size + (3 * num_asc) + 2)) + out_size

def count_params_rnn(in_size, hid_size, out_size):
    return hid_size * (hid_size + in_size + 1) + out_size

def count_params_lstm(in_size, hid_size, out_size):
    return 4 * (hid_size ** 2) + (hid_size * ((4 * in_size) + 8 + out_size)) + out_size

def hid_size_glif(num_params, in_size, out_size, num_asc, learnparams):
    if not learnparams:
        return hid_size_rnn(num_params, in_size, out_size)
    a = 1
    b = in_size + out_size + (3 * num_asc) + 2
    c = -1 * num_params + out_size

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def hid_size_rnn(num_params, in_size, out_size):
    a = 1
    b = in_size + 1
    c = -1 * num_params + out_size

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def hid_size_lstm(num_params, in_size, out_size):
    a = 4
    b = (4 * in_size) + out_size + 8
    c = -1 * num_params + out_size

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
