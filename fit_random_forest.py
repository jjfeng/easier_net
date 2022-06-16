import sys
import time
import argparse
import logging
import joblib

import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def parse_args():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--num-trees", type=int, default=1000)
    parser.add_argument("--num-jobs", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
    parser.add_argument("--log-file", type=str, default="_output/log_rf.txt")
    parser.add_argument("--out-model-file", type=str, default="_output/rf_model.sav")
    args = parser.parse_args()

    assert args.num_classes != 1

    return args


def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    np.random.seed(args.seed)
    logging.info(args)

    # Load data
    dataset_dict = np.load(args.data_file, allow_pickle=True)
    x = dataset_dict["x"]
    y = dataset_dict["y"]
    n_input = x.shape[1]

    # Fit the model
    if args.num_classes == 0:
        regr = RandomForestRegressor(
            max_depth=args.max_depth,
            random_state=args.seed,
            n_estimators=args.num_trees,
            max_features="sqrt",
            n_jobs=args.num_jobs,
        )
    else:
        y = y.astype(int)
        regr = RandomForestClassifier(
            max_depth=args.max_depth,
            random_state=args.seed,
            n_estimators=args.num_trees,
            max_features="sqrt",
            n_jobs=args.num_jobs,
        )
    regr.fit(x, y.ravel())

    # Logging information
    logging.info("FEATURE IMPORT")
    sort_idxs = np.argsort(regr.feature_importances_)
    for idx in sort_idxs[-50:]:
        importance = regr.feature_importances_[idx]
        logging.info("%d: %f", idx, importance)

    output = regr.predict(x)
    if args.num_classes == 0:
        print(output.shape)
        print(y.shape)
        print(output.flatten()[:5])
        print(y.flatten()[:5])
        sq_errs = np.power(output.flatten() - y.flatten(), 2)
        print("median", np.median(sq_errs))
        print("mean", np.mean(sq_errs))
        mse_loss = np.mean(sq_errs)
        logging.info(f"train MSE LOSS {mse_loss}")
    else:
        output = regr.predict_log_proba(x)
        log_prob_class = np.array([output[i, y[i]] for i in range(y.shape[0])])
        logging.info("neg log lik %f", -np.mean(log_prob_class))
        print("neg log lik %f" % -np.mean(log_prob_class))

    joblib.dump(regr, args.out_model_file)


if __name__ == "__main__":
    main(sys.argv[1:])
