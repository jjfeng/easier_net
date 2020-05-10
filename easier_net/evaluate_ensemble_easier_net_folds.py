import argparse
import sys
import json
import pickle

import numpy as np
import pandas as pd

import torch

from evaluate_siernet_folds import eval_fold_models
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
    parser.add_argument("--sample-out-model-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()

    return args


def load_easier_nets(model_file):
    meta_state_dict = torch.load(model_file)
    all_models = []
    for fold_dicts in meta_state_dict["state_dicts"]:
        init_models = []
        for fold_state_dict in fold_dicts:
            model = SierNet(
                n_layers=meta_state_dict["n_layers"],
                n_input=meta_state_dict["n_inputs"],
                n_hidden=meta_state_dict["n_hidden"],
                n_out=meta_state_dict["n_out"],
                input_filter_layer=meta_state_dict["input_filter_layer"],
            )
            model.load_state_dict(fold_state_dict)
            init_models.append(model)
        all_models.append(init_models)
    return all_models, meta_state_dict


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
    all_models, meta_state_dict = load_easier_nets(args.model_file)
    all_losses = []
    for fold_idx, fold_dict in enumerate(fold_idx_dicts):
        test_x = x[fold_dict["test"]]
        test_y = y[fold_dict["test"]]
        fold_models = [seed_fold_models[fold_idx] for seed_fold_models in all_models]
        empirical_loss = eval_fold_models(test_x, test_y, fold_models)
        all_losses.append(empirical_loss)
    avg_loss = np.mean(all_losses)

    # Store a sample model
    if args.sample_out_model_file is not None:
        meta_state_dict["state_dicts"] = meta_state_dict["state_dicts"][0]
        torch.save(meta_state_dict, args.sample_out_model_file)

    # Store the ensemble results
    meta_state_dict.pop("state_dicts", None)
    meta_state_dict.pop("weight", None)
    meta_state_dict["cv_loss"] = float(avg_loss)
    meta_state_dict["seed_losses"] = list(map(float, all_losses))
    print(meta_state_dict)
    json.dump(meta_state_dict, open(args.out_file, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
