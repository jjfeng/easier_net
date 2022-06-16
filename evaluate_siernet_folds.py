import argparse
import sys
import json
import pickle

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import torch

from network import SierNet


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("data_file", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--fold-idxs-file", type=str, default=None)
    parser.add_argument("--out-file", type=str, default="_output/eval.json")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def eval_fold_models(
    test_x: np.ndarray, test_y: np.ndarray, fold_models: list
) -> float:
    is_regression = fold_models[0].is_regression
    fold_outputs = np.array(
        [
            model.predict(test_x) if is_regression else model.predict_log_proba(test_x)
            for model in fold_models
        ]
    )

    if is_regression:
        fold_outputs = np.mean(fold_outputs, axis=0)
        assert fold_outputs.size == test_y.size
        empirical_loss = np.mean(np.power(fold_outputs.flatten() - test_y.flatten(), 2))
    else:
        # Average across logit outputs
        fold_outputs = logsumexp(fold_outputs, axis=0) - np.log(len(fold_models))
        log_prob_class = np.array(
            [fold_outputs[i, test_y[i]] for i in range(test_y.shape[0])]
        ).flatten()
        empirical_loss = -np.mean(log_prob_class)
    return empirical_loss


def load_sier_net(model_file):
    meta_state_dict = torch.load(model_file)
    # Get all the models for the first seed
    fold_dicts = meta_state_dict["state_dicts"][0]
    fold_models = []
    for fold_state_dict in fold_dicts:
        model = SierNet(
            n_layers=meta_state_dict["n_layers"],
            n_input=meta_state_dict["n_inputs"],
            n_hidden=meta_state_dict["n_hidden"],
            n_out=meta_state_dict["n_out"],
            input_filter_layer=meta_state_dict["input_filter_layer"],
        )
        model.load_state_dict(fold_state_dict)
        fold_models.append(model)
    return fold_models, meta_state_dict


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(args.seed)

    # Load data
    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]
    y = dataset_dict["y"]
    true_y = dataset_dict["true_y"]

    # Load folds
    with open(args.fold_idxs_file, "rb") as f:
        fold_idx_dicts = pickle.load(f)
        num_folds = len(fold_idx_dicts)

    # Load models and evaluate them on folds, take the average
    fold_models, meta_state_dict = load_sier_net(args.model_file)
    all_losses = []
    for fold_idx, fold_dict in enumerate(fold_idx_dicts[: len(fold_models)]):
        test_x = x[fold_dict["test"]]
        test_y = y[fold_dict["test"]]
        # eval loss for singleton
        empirical_loss = eval_fold_models(test_x, test_y, [fold_models[fold_idx]])
        all_losses.append(empirical_loss)
    avg_loss = np.mean(all_losses)

    # Store the ensemble results
    meta_state_dict.pop("state_dicts", None)
    meta_state_dict.pop("weight", None)
    meta_state_dict["cv_loss"] = float(avg_loss)
    meta_state_dict["seed_losses"] = list(map(float, all_losses))
    print(meta_state_dict)
    json.dump(meta_state_dict, open(args.out_file, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
