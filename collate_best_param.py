import json
import glob
import argparse
import sys
import logging
import pickle

from scipy.special import logsumexp
import numpy as np
import itertools

import torch

from network import SierNet
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
    parser.add_argument("--template", type=str)
    parser.add_argument("--full-tree-pens", type=str)
    parser.add_argument("--input-pens", type=str)
    parser.add_argument("--layers", type=str)
    parser.add_argument("--log-file", type=str, default="_output/best_option.txt")
    parser.add_argument("--out-file", type=str, default="_output/best_option.json")
    parser.set_defaults()
    args = parser.parse_args()

    args.full_tree_pens = process_params(args.full_tree_pens, float)
    args.input_pens = process_params(args.input_pens, float)
    args.layers = process_params(args.layers, int)

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    option_res = {}
    for layers in args.layers:
        for full_tree_pen in args.full_tree_pens:
            for input_pen in args.input_pens:
                res_file = args.template % (layers, input_pen, full_tree_pen)
                try:
                    with open(res_file, "r") as f:
                        res = json.load(f)
                except FileNotFoundError as e:
                    print("NOT FOUND", res_file)
                    continue
                opt_param = tuple([(k, v) for k, v in res.items() if "loss" not in k])
                option_res[opt_param] = res["cv_loss"]

    best_option = None
    best_loss = np.inf
    for model_opt, emp_loss in option_res.items():
        logging.info(f"Ensemble choice {model_opt}, loss {emp_loss}")
        print(f"Ensemble choice {model_opt}, loss {emp_loss}")
        if emp_loss < best_loss:
            best_option = model_opt
            best_loss = emp_loss
    logging.info(f"Best choice {best_option}, loss {best_loss}")
    print(f"Best choice {best_option}, loss {best_loss}")

    # Convert tuple to dict
    best_option_dict = {k: v for k, v in best_option}
    print("BEST OPT", best_option_dict)

    # Save results
    json.dump(best_option_dict, open(args.out_file, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
