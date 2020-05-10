"""
Fit ResNet with proper penalization
"""
import argparse
import sys
import logging
import json
import pickle

import numpy as np
from scipy.stats import bernoulli
import itertools

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import GridSearchCV

from .plain_nnet import PlainNetEstimator
from .common import process_params


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-hidden", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--max-prox-iters", type=int, default=0)
    parser.add_argument("--bootstrap", action="store_true", default=False)
    parser.add_argument("--input-pen", type=float, default=0)
    parser.add_argument("--full-tree-pens", type=str, default="0.001")
    parser.add_argument("--batch-obs-size", type=int, default=5000)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--input-filter-layer", action="store_true", default=False)
    parser.add_argument("--k-fold", type=int, default=None)
    parser.add_argument("--fold-idxs-file", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--log-file", type=str, default="_output/log_nn.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/nn.pt")
    parser.set_defaults(bootstrap=False)
    args = parser.parse_args()

    args.full_tree_pens = process_params(args.full_tree_pens, float)

    assert args.num_classes != 1
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """
    Load data
    """
    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]
    orig_y = dataset_dict["y"]
    n_inputs = x.shape[1]
    n_out = 1 if args.num_classes == 0 else args.num_classes
    n_obs = x.shape[0]

    if args.fold_idxs_file is not None:
        with open(args.fold_idxs_file, "rb") as f:
            fold_idx_dict = pickle.load(f)
            fold_idxs = [(a["train"], a["test"]) for a in fold_idx_dict]

    """
    Bootstrap sample
    """
    if args.bootstrap:
        not_same_uniq_classes = True
        while not_same_uniq_classes:
            chosen_idxs = np.random.choice(n_obs, size=n_obs, replace=True)
            x = x[chosen_idxs]
            y = orig_y.flatten()[chosen_idxs].reshape((-1, 1))
            if args.num_classes >= 2:
                not_same_uniq_classes = np.unique(y).size != np.unique(orig_y).size
            else:
                not_same_uniq_classes = False
    else:
        y = orig_y

    """
    Fit
    """
    plain_nn_est = PlainNetEstimator(
        n_inputs=n_inputs,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_out=n_out,
        full_tree_pen=args.full_tree_pens[0],
        input_pen=args.input_pen,
        max_iters=args.max_iters,
        max_prox_iters=args.max_prox_iters,
        batch_size=(n_obs // args.num_batches + 1)
        if args.num_batches is not None
        else args.batch_obs_size,
        num_classes=args.num_classes,
        # Weight classes by inverse of their observed ratios. Trying to balance classes
        weight=n_obs / (args.num_classes * np.bincount(y.flatten()))
        if args.num_classes >= 2
        else None,
        dropout=args.dropout,
        input_filter_layer=args.input_filter_layer,
    )
    if len(args.full_tree_pens) == 1:
        plain_nn_est.fit(x, y)
        net = plain_nn_est.net
    else:
        tune_parameters = [
            {
                "n_inputs": [n_inputs],
                "n_layers": [args.n_layers],
                "n_hidden": [args.n_hidden],
                "n_out": [n_out],
                "full_tree_pen": args.full_tree_pens,
                "input_pen": [args.input_pen],
                "max_iters": [args.max_iters],
                "max_prox_iters": [args.max_prox_iters],
                "batch_size": [args.batch_obs_size],
                "num_classes": [args.num_classes],
                "dropout": [args.dropout],
                "input_filter_layer": [args.input_filter_layer],
            }
        ]

        cv_plain_nn = GridSearchCV(
            plain_nn_est,
            tune_parameters,
            cv=args.k_fold if args.fold_idxs_file is None else fold_idxs,
            verbose=True,
            refit=True,
            n_jobs=args.n_jobs,
        )
        cv_plain_nn.fit(x, y)

        logging.info(cv_plain_nn.cv_results_["mean_test_score"])
        logging.info(cv_plain_nn.cv_results_["params"])
        logging.info(cv_plain_nn.best_params_)

        net = cv_plain_nn.best_estimator_.net
        plain_nn_est = cv_plain_nn.best_estimator_

    meta_state_dict = plain_nn_est.get_params()
    meta_state_dict["state_dict"] = net.state_dict()
    torch.save(meta_state_dict, args.out_model_file)


if __name__ == "__main__":
    main(sys.argv[1:])
