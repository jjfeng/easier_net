"""
Fit ResNet with proper penalization
"""
import time
import argparse
import copy
import sys
import logging
import json
from joblib import Parallel, delayed
import pickle
import itertools

import numpy as np
from scipy.stats import bernoulli
from sklearn.base import clone
import itertools

import torch
from torch.utils.data import DataLoader

import sier_net
import easier_net
import common


def parse_args(args):
    """parse command line arguments"""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--data-file", type=str, default="_output/data.npz")

    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Number of classes in classification. Should be zero if doing regression",
    )
    parser.add_argument("--input-filter-layer", action="store_true", default=True)
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=10, help="Number of hidden nodes per layer"
    )
    parser.add_argument(
        "--num-inits",
        type=int,
        default=1,
        help="Determines the number of initializations, which corresponds to the size of the ensemble",
    )
    parser.add_argument(
        "--max-iters", type=int, default=40, help="Number of Adam epochs"
    )
    parser.add_argument(
        "--max-prox-iters",
        type=int,
        default=0,
        help="Number of batch proximal gradient descent epochs (after running Adam)",
    )
    parser.add_argument(
        "--full-tree-pen",
        type=float,
        default=0.001,
        help="Corresponds to lambda2 in the paper",
    )
    parser.add_argument(
        "--input-pen", type=float, default=0, help="Corresponds to lambda1 in the paper"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="How many mini-batches to use when using Adam",
    )
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--model-fit-params-file",
        type=str,
        help="A json file that specifies what the hyperparameters are. If given, this will override the arguments passed in.",
    )
    # TODO: DELETE?
    parser.add_argument("--log-file", type=str, default="_output/log_nn.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/nn.pt")
    args = parser.parse_args()

    if args.model_fit_params_file is not None:
        with open(args.model_fit_params_file, "r") as f:
            model_params = json.load(f)
            args.full_tree_pen = model_params["full_tree_pen"]
            args.input_pen = model_params["input_pen"]
            args.input_filter_layer = model_params["input_filter_layer"]
            args.n_layers = model_params["n_layers"]
            args.n_hidden = model_params["n_hidden"]

    assert args.num_classes != 1
    assert args.input_filter_layer

    return args


def _fit(
    estimator,
    X,
    y,
    train,
    seed: int = 0,
) -> list:
    torch.manual_seed(seed)
    X_train = X[train]
    y_train = y[train]

    my_estimator = clone(estimator)
    my_estimator.fit(
        X_train, y_train  
    )
    return my_estimator


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    st_time = time.time()

    """
    Load data
    """
    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]
    y = dataset_dict["y"]

#TODO: remove this + args in base estimator upon confirmation
    n_inputs = x.shape[1]
    n_out = 1 if args.num_classes == 0 else args.num_classes
    n_obs = x.shape[0]

    """
    Fit EASIER-net
    """
    print("Fitting EASIER-net")
    base_estimator = easier_net.EasierNetEstimator(
        # n_inputs=n_inputs,
        input_filter_layer=args.input_filter_layer,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        # n_out=n_out,
        full_tree_pen=args.full_tree_pen,
        input_pen=args.input_pen,
        num_batches=args.num_batches,
        # batch_size=(n_obs // args.num_batches + 1),
        num_classes=args.num_classes,
        max_iters=args.max_iters,
        max_prox_iters=args.max_prox_iters,
    )

    all_estimators = base_estimator.fit(
        x[np.arange(x.shape[0])], y[np.arange(x.shape[0])]
    )
    all_estimators.write_model(args.out_model_file)


    logging.info("FINAL STRUCT idx 0")

    logging.info("complete")
    logging.info("TIME %f", time.time() - st_time)


if __name__ == "__main__":
    main(sys.argv[1:])
