# EASIER-net

Python code for fitting EASIER-nets and reproducing all results from the paper.
The python code uses [PyTorch](https://pytorch.org/).

## Quick-start

Setup a python virtual environment (code runs for python 3.6) with the appropriate packages from `requirements.txt`.

Simulate data using `generate_data.py` or load your own into a `npz` format with `x` and `y` attributes.

To fit an EASIER-net, run
```
python fit_easier_net.py --num-classes <NUM_CLASSES> --n-layers <N_LAYERS> --n-hidden <N_HIDDEN> --num-inits <NUM_INITS> --input-pen <INPUT_PEN> --full-tree-pen <FULL_TREE_PEN> --max-iters <MAX_ITERS> --max-prox-iters <MAX_PROX_ITERS> --num-batches <NUM_BATCHES> --out-model-file <OUT_MODEL_FILE>
```
where:
* `NUM_CLASSES` should be 0 if doing regression and `NUM_CLASSES` should be the number of classes if doing binary/multi-classification
* `N_LAYERS` is the number of hidden layers
* `N_HIDDEN` is the number of hidden nodes per layer
* `NUM_INITS` is the size of the ensemble
* `INPUT_PEN` specifies $\lambda_1$ in the paper
* `FULL_TREE_PEN` specifies $\lambda_2$ in the paper
* `MAX_ITERS` is the number of epochs to run Adam
* `MAX_PROX_ITERS` is the number of epochs to run batch proximal gradient descent
* `NUM_BATCHES` is the number of mini-batches when running Adam (default 3) 
* `OUT_MODEL_FILE` contains the fitted model

To perform cross-validation, one should run separate `fit_easier_net.py` scripts for each candidate penalty parameter values.
Then select the best penalty parameter values using `collate_best_param.py`.


## Reproducing results

Make sure you have [scons](https://scons.org/) downloaded.
To reproduce results, run `scons <folder_name>`.
The folders associated with each empirical experiment is given below:

* Simulation study of deconstructed EASIER-nets: `simulation_deconstruct`.
* Simulation study of variable selection: `simulation_support_prob`.
* Comparison on UCI datasets: `uci_data_classification` and `uci_data_regression`
   + Note that you run `load_uci_classification_data.py` and `load_uci_regression_data.py` to first load the data.
