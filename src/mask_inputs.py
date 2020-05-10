import argparse
import sys
import logging
import pickle

import numpy as np
import itertools

from common import process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("masked_inputs", type=str)
    parser.add_argument("--in-data-file", type=str, default=None)
    parser.add_argument("--out-data-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()
    args.masked_inputs = process_params(args.masked_inputs, int)
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    # Load data
    dataset_dict = np.load(args.in_data_file)
    x = dataset_dict["x"]
    y = dataset_dict["y"]
    true_y = dataset_dict["true_y"]

    # Delete the masked inputs
    x = np.delete(x, args.masked_inputs, 1)

    # Save masked data
    np.savez_compressed(args.out_data_file, x=x, y=y, true_y=true_y)


if __name__ == "__main__":
    main(sys.argv[1:])
