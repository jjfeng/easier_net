# EASIER-net

Feng, Jean, and Noah Simon. 2022. “Ensembled Sparse‐input Hierarchical Networks for High‐dimensional Datasets.” Statistical Analysis and Data Mining, March. https://doi.org/10.1002/sam.11579.

Python code for fitting EASIER-nets and reproducing all results from the paper.
The python code uses [PyTorch](https://pytorch.org/).

R code for fitting EASIER-net is available at https://github.com/jjfeng/easier_net_R.

## Quick-start

Setup a python virtual environment (code runs for python 3.6) with the appropriate packages from `requirements.txt`.

Simulate data using by following the tutorial notebook or load your own into a `npz` format with `x` and `y` attributes. You may also perform GridSearchCV by following the tutorial.

To fit an EASIER-net, run
```
python fit_easier_net.py --n-estimators <N_ESTIMATORS> --input-filter-layer <INPUT_FILTER_LAYER> --n-layers <N_LAYERS> --n-hidden <N_HIDDEN> --input-pen <INPUT_PEN> --full-tree-pen <FULL_TREE_PEN> --batch-size <BATCH_SIZE> --num-classes <NUM_CLASSES>  --weight <WEIGHT> --max-iters <MAX_ITERS> --max-prox-iters <MAX_PROX_ITERS> --model-fit-params-file <MODEL_FIT_PARAMS_FILE>
```
where:
* `N_ESTIMATORS` should be size of ensemble; the number of SIER-nets being ensembled.
* `INPUT_FILTER_LAYER` is whether to scale the inputs by parameter β
* `N_LAYERS` is the number of hidden layers
* `N_HIDDEN` is the number of hidden nodes per layer
* `INPUT_PEN` specifies $\lambda_1$ in the paper; controls the input sparsity
* `FULL_TREE_PEN` specifies $\lambda_2$ in the paper; controls the number of active layers and hidden nodes
* `BATCH_SIZE` specifies the size of the mini-batches for Adam
* `NUM_CLASSES` should be 0 if doing regression and `NUM_CLASSES` should be the number of classes if doing binary/multi-classification
* `WEIGHT` is a list of weights for the classes
* `MAX_ITERS` is the number of epochs to run Adam
* `MAX_PROX_ITERS` is the number of epochs to run batch proximal gradient descent
* `MODEL_FIT_PARAMS_FILE` is a json file that specifies what the hyperparameters are. If given, this will override the arguments passed in.

To perform cross-validation, one should run separate `fit_easier_net.py` scripts for each candidate penalty parameter values.
Then select the best penalty parameter values using `collate_best_param.py`.
