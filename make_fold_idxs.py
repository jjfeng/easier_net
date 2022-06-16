import argparse
import sys
import logging
import pickle

import numpy as np
import itertools

from sklearn.model_selection import KFold
from common import process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--k-fold", type=int, default=3)
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
    parser.add_argument("--out-file", type=str, default="_output/folds.pkl")
    parser.set_defaults()
    args = parser.parse_args()
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    """
    Load data
    """
    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]

    """
    Make K folds
    """
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    train_test_splits = [{"train": train, "test": test} for train, test in kf.split(x)]
    with open(args.out_file, "wb") as f:
        pickle.dump(train_test_splits, f)


if __name__ == "__main__":
    main(sys.argv[1:])
