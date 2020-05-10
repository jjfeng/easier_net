import argparse
import sys
import logging
import pickle

import numpy as np
import itertools

import xgboost as xgb
import torch
import torch.nn as nn

from data_generator import DataGenerator

from sklearn.model_selection import GridSearchCV

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
    parser.add_argument("--num-rounds", type=str, default="200")
    parser.add_argument("--max-depths", type=str, default="4")
    parser.add_argument(
        "--reg-lambda",
        type=float,
        help="L2 regularization term on weights",
        default=0.1,
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        help="L1 regularization term on weights",
        default=0.05,
    )
    parser.add_argument("--k-fold", type=int, default=None)
    parser.add_argument("--fold-idxs-file", type=str, default=None)
    parser.add_argument(
        "--num-classes", type=int, default=0,
    )
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--log-file", type=str, default="_output/log_xgboost.txt")
    parser.add_argument(
        "--out-model-file", type=str, default="_output/xgboost_model.json"
    )
    parser.set_defaults()
    args = parser.parse_args()

    assert args.num_classes != 1
    args.max_depths = process_params(args.max_depths, int)
    args.num_rounds = process_params(args.num_rounds, int)

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    # Load data
    dataset_dict = np.load(args.data_file, allow_pickle=True)
    x = dataset_dict["x"]
    y = dataset_dict["y"].flatten() if args.num_classes >= 2 else dataset_dict["y"]

    if args.fold_idxs_file is not None:
        with open(args.fold_idxs_file, "rb") as f:
            fold_idx_dict = pickle.load(f)
            fold_idxs = [(a["train"], a["test"]) for a in fold_idx_dict]

    # Train the xgb model
    if args.num_classes >= 2:
        bst = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.01,
            n_classifiers=args.num_rounds[0],
            max_depth=args.max_depths[0],
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            n_jobs=args.n_jobs,
        )
    elif args.num_classes == 0:
        bst = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_classifiers=args.num_rounds[0],
            max_depth=args.max_depths[0],
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            n_jobs=args.n_jobs,
        )

    if len(args.num_rounds) * len(args.max_depths) == 1:
        bst.fit(x, y)
    else:
        tune_parameters = [
            {
                "reg_alpha": [args.reg_alpha],
                "reg_lambda": [args.reg_lambda],
                "n_estimators": args.num_rounds,
                "max_depth": args.max_depths,
            }
        ]

        cv_bst = GridSearchCV(
            bst,
            tune_parameters,
            cv=args.k_fold if args.fold_idxs_file is None else fold_idxs,
            verbose=True,
            refit=True,
            n_jobs=args.n_jobs,
        )
        cv_bst.fit(x, y)
        logging.info("mean test scores %s", str(cv_bst.cv_results_["mean_test_score"]))
        logging.info(f"BEST {cv_bst.best_params_}")
        bst = cv_bst.best_estimator_

    # Print some stuff
    if args.num_classes >= 2:
        output = np.log(bst.predict_proba(x))
        log_prob_class = [output[i, y[i]] for i in range(y.shape[0])]
        logging.info("neg log lik %f", -np.mean(log_prob_class))
        print("neg log lik %f" % -np.mean(log_prob_class))
    else:
        output = bst.predict(x)
        print(output.shape, y.shape)
        mse_loss = np.mean(np.power(output.flatten() - y.flatten(), 2))
        logging.info("mse loss %f", mse_loss)
        print("train MSE LOSS", mse_loss)

    bst.save_model(args.out_model_file)


if __name__ == "__main__":
    main(sys.argv[1:])
