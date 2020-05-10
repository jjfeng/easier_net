import sys
import pickle
import os
import joblib
import argparse
import logging

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression


def parse_args():
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k-fold", type=int, default=3)
    parser.add_argument("--fold-idxs-file", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="_output/data.npz")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--log-file", type=str, default="_output/lasso.log")
    parser.add_argument("--out-model-file", type=str, default="_output/lasso_model.sav")
    parser.add_argument("--scratch", type=str, default="_output/scratch")
    args = parser.parse_args()

    assert args.num_classes != 1

    return args


def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    logging.info(args)

    # Load data
    dataset_dict = np.load(args.data_file, allow_pickle=True)
    x = dataset_dict["x"]
    y = dataset_dict["y"]
    n_input = x.shape[1]

    if args.fold_idxs_file is not None:
        with open(args.fold_idxs_file, "rb") as f:
            fold_idx_dict = pickle.load(f)
            fold_idxs = [(a["train"], a["test"]) for a in fold_idx_dict]

    if args.num_classes == 0:
        lasso = Lasso(random_state=args.seed, max_iter=10000)
        alphas = np.power(10.0, np.arange(2, -2, -0.3))  # step -0.1
        tuned_parameters = [{"alpha": alphas}]
    elif args.num_classes == 2:
        lasso = LogisticRegression(random_state=args.seed, max_iter=10000, penalty="l1")
        Cs = np.power(10.0, np.arange(4, -3, -0.3))  # step -0.1
        tuned_parameters = [{"C": Cs}]
    elif args.num_classes > 2:
        lasso = LogisticRegression(
            random_state=args.seed,
            max_iter=10000,
            penalty="l1",
            multi_class="multinomial",
            solver="saga",
        )
        Cs = np.power(10.0, np.arange(4, -3, -0.3))  # step -0.1
        tuned_parameters = [{"C": Cs}]

    clf = GridSearchCV(
        lasso,
        tuned_parameters,
        cv=args.k_fold if args.fold_idxs_file is None else fold_idxs,
        verbose=True,
        refit=True,
        n_jobs=args.n_jobs,
    )
    clf.fit(x, y.ravel())
    model = clf.best_estimator_

    logging.info(clf.cv_results_["mean_test_score"])
    logging.info(clf.cv_results_["params"])
    logging.info(clf.best_params_)
    logging.info(clf.best_estimator_.coef_)
    logging.info("num nonzero entries %d", np.count_nonzero(clf.best_estimator_.coef_))

    if args.num_classes == 0:
        output = model.predict(x)
        mse_loss = np.mean(np.power(output - y, 2))
        logging.info(f"train MSE LOSS {mse_loss}")
    else:
        output = model.predict_log_proba(x)
        log_prob_class = [output[i, y[i]] for i in range(y.shape[0])]
        logging.info("neg log lik %f", -np.mean(log_prob_class))
        print("neg log lik %f" % -np.mean(log_prob_class))

    joblib.dump(model, args.out_model_file)


if __name__ == "__main__":
    main(sys.argv[1:])
