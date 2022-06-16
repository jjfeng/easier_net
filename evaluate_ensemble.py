import argparse
import sys
import logging
import joblib
import glob

import numpy as np
import pandas as pd

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from evaluate_model import do_model_inference


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("data_file", type=str)
    parser.add_argument("model_files", type=str)
    parser.add_argument(
        "model_loader_class", type=str, choices=["plain_nnet", "resnet", "easier_net"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="_output/eval.txt")
    parser.add_argument("--out-file", type=str, default="_output/eval.csv")
    parser.set_defaults()
    args = parser.parse_args()

    args.model_files = glob.glob(args.model_files)
    assert len(args.model_files) > 0
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
    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]
    y = dataset_dict["y"]
    true_y = dataset_dict["true_y"]

    # Load model
    # Note that we assume the ensemble contains all of the same model class
    all_outputs = []
    supports = []
    for model_file in args.model_files:
        print(model_file)
        outputs, model = do_model_inference(
            args.model_loader_class, model_file, args.num_classes, x
        )
        support = np.where(model.support() > 0)[0]
        logging.info(support)
        logging.info("SUPPORT LEN %d", support.size)
        supports.append(support)
        if args.num_classes == 0:
            all_outputs.append(outputs)
        else:
            all_outputs.append(np.exp(outputs))

    # Look at the support
    print(supports[0])
    intersected_support = set(supports[0]).intersection(*[set(s) for s in supports])
    print(f"INTERSECT {intersected_support}, tot count {len(intersected_support)}")
    logging.info(f"INTERSECT {intersected_support}")

    counts = np.zeros(x.shape[1])
    for support in supports:
        for i in support:
            counts[i] += 1
    majority_support = np.where(counts > len(args.model_files) * 0.5)[0]
    print(f"majority support {majority_support}, tot count: {len(majority_support)}")
    logging.info(
        f"majority support {majority_support}, tot count: {len(majority_support)}"
    )

    # Look at performance
    all_outputs = np.array(all_outputs)
    outputs = np.mean(all_outputs, axis=0)
    print(all_outputs[:, 0, :])
    if args.num_classes == 0:
        assert outputs.size == true_y.size
        empirical_loss = np.mean(np.power(outputs.flatten() - true_y.flatten(), 2))
        logging.info(f"test MSE LOSS {empirical_loss}")
        print(f"test MSE LOSS {empirical_loss}")
    else:
        outputs = np.log(outputs)
        log_prob_class = np.array(
            [outputs[i, y[i]] for i in range(y.shape[0])]
        ).flatten()
        print("median", np.median(log_prob_class))
        print("mean", np.mean(log_prob_class))
        print("LOG PROB SORT", np.sort(log_prob_class)[:10])
        empirical_loss = -np.mean(log_prob_class)
        print(f"neg log lik {empirical_loss}")
        logging.info(f"neg log lik {empirical_loss}")
    loss_df = pd.DataFrame(
        {
            "model_class": [
                model.__class__.__name__ if args.model_name is None else args.model_name
            ],
            "test_loss": [empirical_loss],
        }
    )
    loss_df.to_csv(open(args.out_file, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
