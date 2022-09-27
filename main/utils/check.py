import argparse

def check_nonnegative_int(num):
    num = int(num)
    if num < 0:
        raise argparse.ArgumentTypeError('nonnegative integer expected')
    return num

def check_positive_int(num):
    num = int(num)
    if num <= 0:
        raise argparse.ArgumentTypeError('nonnegative integer expected')
    return num

def check_nonnegative_float(num):
    num = float(num)
    if num < 0:
        raise argparse.ArgumentTypeError('nonnegative float expected')
    return num

def check_positive_float(num):
    num = float(num)
    if num <= 0:
        raise argparse.ArgumentTypeError('nonnegative float expected')
    return num