import argparse

def make_parser():
    parser = argparse.ArgumentParser("Runner")
    parser.add_argument("-m", "--movie_path", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")

    return parser