import math

def count_params_glif(in_size, hid_size, out_size, num_asc, learnparams=True):
    """
    Computes number of parameters used by GLIFR network.
    """
    if not learnparams:
        return (hid_size ** 2) + (hid_size * (in_size + out_size)) + out_size#count_params_rnn(in_size, hid_size, out_size)
    return (hid_size ** 2) + (hid_size * (in_size + out_size + (3 * num_asc) + 2)) + out_size

def count_params_rnn(in_size, hid_size, out_size):
    """
    Computes number of parameters used by RNN network.
    """
    return hid_size * (hid_size + in_size + out_size + 1) + out_size

def count_params_lstm(in_size, hid_size, out_size):
    """
    Computes number of parameters used by LSTM network.
    """
    return 4 * (hid_size ** 2) + (hid_size * ((4 * in_size) + 8 + out_size)) + out_size

def hid_size_glif(num_params, in_size, out_size, num_asc, learnparams):
    """
    Computes number of neurons in GLIFR network to achieve given number of parameters.
    """
    if not learnparams:
        a = 1
        b = in_size + out_size
        c = -1 * num_params + out_size
    else:
        a = 1
        b = in_size + out_size + (3 * num_asc) + 2
        c = -1 * num_params + out_size

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def hid_size_rnn(num_params, in_size, out_size):
    """
    Computes number of neurons in RNN network to achieve given number of parameters.
    """
    a = 1
    b = out_size + in_size + 1
    c = -1 * num_params + out_size

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def hid_size_lstm(num_params, in_size, out_size):
    """
    Computes number of neurons in LSTM network to achieve given number of parameters.
    """
    a = 4
    b = (4 * in_size) + out_size + 8
    c = -1 * num_params + out_size

    return int((-b + math.sqrt((b**2) - (4 * a * c))) / (2 * a))

def count_parameters(model):
    """
    Computes number of parameters in given PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_dict(in_size, out_size, hid_size_rnn):
    nparams = count_params_rnn(in_size, hid_size_rnn, out_size)

    result = {}
    result["rnn"] = hid_size_rnn
    result["glifr_lheta"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=2, learnparams=True)
    result["glifr_lhet"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=0, learnparams=True)
    result["glifr_rheta"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=2, learnparams=True)
    result["glifr_rhet"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=0, learnparams=True)
    result["glifr_fheta"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=2, learnparams=False)
    result["glifr_fhet"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=0, learnparams=False)
    result["glifr_homa"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=2, learnparams=False)
    result["glifr_hom"] = hid_size_glif(num_params=nparams, in_size=in_size, out_size=out_size, num_asc=0, learnparams=False)
    result["lstmn"] = hid_size_lstm(num_params=nparams, in_size=in_size, out_size=out_size)
    print(result)

if __name__ == '__main__':
    print("Sine:")
    print_dict(1, 1, 128)

    print("LMNIST:")
    print_dict(28, 10, 256)
    
    print("PMNIST:")
    print_dict(1, 10, 256)

    print("NMNIST:")
    print_dict(1, 10, 256)