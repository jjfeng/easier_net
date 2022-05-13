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

from sklearn.model_selection import GridSearchCV

import sier_net
import common

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
    # parser.add_argument(
    #     "--fold-idxs-file",
    #     type=str,
    #     default=None,
    #     help="If specified, the code will fit a separate model per fold, in a K-fold CV fashion. This pickle file specifies the indices in each fold. The fold indices should be produced using make_fold_idx.py.",
    # )
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
    # parser.add_argument("--bootstrap", action="store_true", default=False)
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
    parser.add_argument("--log-file", type=str, default="_output/log_nn.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/nn.pt")
    # parser.set_defaults(bootstrap=False)
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
    # max_iters: int = 100,
    # max_prox_iters: int = 100,
    seed: int = 0,
) -> list:
    torch.manual_seed(seed)
    X_train = X[train]
    y_train = y[train]

    my_estimator = clone(estimator)
    my_estimator.fit(
        X_train, y_train #, max_iters=max_iters, max_prox_iters=max_prox_iters
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

    n_inputs = x.shape[1]
    n_out = 1 if args.num_classes == 0 else args.num_classes
    n_obs = x.shape[0]

    # """
    # Bootstrap sample
    # """
    # if args.bootstrap:
    #     not_same_uniq_classes = True
    #     while not_same_uniq_classes:
    #         chosen_idxs = np.random.choice(n_obs, size=n_obs, replace=True)
    #         x = x[chosen_idxs]
    #         y = orig_y.flatten()[chosen_idxs].reshape((-1, 1))
    #         if args.num_classes >= 2:
    #             not_same_uniq_classes = np.unique(y).size != np.unique(orig_y).size
    #         else:
    #             not_same_uniq_classes = False
    # else:
    #     y = orig_y 

    """
    Fit EASIER-net
    """
    print("Fitting EASIER-net")
    base_estimator = sier_net.SierNetEstimator(
        n_inputs=n_inputs,
        input_filter_layer=args.input_filter_layer,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_out=n_out,
        full_tree_pen=args.full_tree_pen,
        input_pen=args.input_pen,
        batch_size=(n_obs // args.num_batches + 1),
        num_classes=args.num_classes,
        max_iters=args.max_iters,
        max_prox_iters=args.max_prox_iters,
        # Weight classes by inverse of their observed ratios. Trying to balance classes
        weight=n_obs / (args.num_classes * np.bincount(y.flatten()))
        if args.num_classes >= 2
        else None,
    )
    # if args.fold_idxs_file is not None:
    #     with open(args.fold_idxs_file, "rb") as f:
    #         fold_idx_dict = pickle.load(f)
    #         num_folds = len(fold_idx_dict)

    #     parallel = Parallel(n_jobs=args.n_jobs, verbose=True, pre_dispatch=args.n_jobs)
    #     all_estimators = parallel(
    #         delayed(_fit)(
    #             base_estimator, #single sier net
    #             x,
    #             y,
    #             train=fold_idx_dict[fold_idx]["train"],
    #             max_iters=args.max_iters,
    #             max_prox_iters=args.max_prox_iters,
    #             seed=args.seed + num_folds * init_idx + fold_idx,
    #         )
    #         for fold_idx, init_idx in itertools.product(
    #             range(num_folds), range(args.num_inits)
    #         )
    #     )

    #     # Just printing things from the first fold
    #     logging.info(f"sample estimator 0 fold 0")
    #     all_estimators[0].net.get_net_struct()

    #     assert (num_folds * args.num_inits) == len(all_estimators)

    #     meta_state_dict = all_estimators[0].get_params()
    #     meta_state_dict["state_dicts"] = [
    #         [None for _ in range(num_folds)] for _ in range(args.num_inits)
    #     ]
    #     for (fold_idx, init_idx), estimator in zip(
    #         itertools.product(range(num_folds), range(args.num_inits)), all_estimators
    #     ):
    #         meta_state_dict["state_dicts"][init_idx][
    #             fold_idx
    #         ] = estimator.net.state_dict()
    #     torch.save(meta_state_dict, args.out_model_file)
    # else:
    all_estimators = [
        _fit(
            base_estimator,
            x,
            y,
            train=np.arange(x.shape[0]),
            # max_iters=args.max_iters,
            # max_prox_iters=args.max_prox_iters,
            seed=args.seed + init_idx,
        )
        for init_idx in range(args.num_inits)
    ]
    meta_state_dict = all_estimators[0].get_params()
    meta_state_dict["state_dicts"] = [
        estimator.net.state_dict() for estimator in all_estimators
    ]
    torch.save(meta_state_dict, args.out_model_file)

    logging.info("FINAL STRUCT idx 0")
    all_estimators[0].net.get_net_struct()

logging.info("complete")
logging.info("TIME %f", time.time() - st_time)


if __name__ == "__main__":
    main(sys.argv[1:])
