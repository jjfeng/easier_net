from ensurepip import bootstrap
# from imblearn import RandomUnderSampler
import numpy as np

import pickle
import sier_net #technically should be from sier_net import SierNetEstimator

class EasierNetEstimator():
    """ This is the Ensembling by Averaging Sparse-Input Hierarchical Networks(EASIER-NET)"""
    #i think my init needs a seed
    def __init__(
        self, #num inits >=5
        # seed=12,
        # data_file="_output/data.npz",
        num_classes=0,
        input_filter_layer=True,
        n_layers=2,
        n_hidden=10,
        num_inits=5,       #change this to n_estimators
        max_iters=40,
        max_prox_iters=0,
        bootstrap=False,
        full_tree_pen=0.001,
        input_pen=0,
        num_batches=3,
        n_jobs=16,
        # model_fit_params_file=None,
        # log_file="_output/log_nn.txt",
        # out_model_file="_output/nn.pt",
        # fold_idxs_file=None,
    ):
        # self._estimator_type="classifier"
        self.num_classes=num_classes
        self.input_filter_layer=input_filter_layer
        self.n_layers=n_layers
        self.n_hidden=n_hidden
        self.num_inits=num_inits
        self.max_iters=max_iters
        self.max_prox_iters=max_prox_iters
        self.bootstrap=bootstrap
        self.full_tree_pen=full_tree_pen
        self.input_pen=input_pen
        self.num_batches=num_batches
        self.n_jobs=n_jobs

        self.estimators=self._generate_estimators()
        #not sure what to do with files (4)

        assert num_classes != 1
        assert input_filter_layer
        
        # if fold_idxs_file is not None:
        #     with open(fold_idxs_file, "rb") as f:
        #         fold_idx_dict = pickle.load(f)
        #         num_folds = len(fold_idx_dict)
    
#copied from sier_net, not yet edited
    def _generate_estimators(self):
        estimators=[]
        for i in range(self.num_inits):
            est=clone(self.base_estimator) #create multiple siernet estimators instead

            est.random_state=i


            est.num_classes=self.num_classes
            est.input_filter_layer=self.input_filter_layer
            est.n_hidden= self.n_hidden
            est.num_inits=self.num_inits
            est.max_iters=self.max_iters
            est.max_prox_iters=self.max_prox_iters
            est.bootstrap=self.bootstrap
            est.full_tree_pen=self.full_tree_pen
            est.input_pen=self.input_pen
            est.num_batches=self.num_batches
            est.n_jobs=self.n_jobs

            #here i want to append each sier net.. 
            estimators.append(est)

        return estimators

    def write_model():
        #write a model

    def load_model(model_fit_params_file):
        #load in each of the state dicts


    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    #     max_iters: int = 100,
    #     max_prox_iters: int = 100,
    #     state_dict: dict = None,
    ) -> None:

        for est in self.estimators:
            est.fit(x, y)
       
        return self 

#score and predict just copied from sier_net?
#refer to line 100 
#write for loop of all the outputs / predictions, then take an average, then apply same scoring function
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        torch_y = (
            torch.Tensor(y)
            if self.num_classes == 0
            else torch.from_numpy(y.astype(int))[:, 0]
        )
        return -self.score_criterion(self.net(torch.Tensor(x)), torch_y).item()

    def predict(self, x: np.ndarray) -> np.ndarray: #another for loop + average, output class label if classication 
        output = self.net(torch.Tensor(x)).detach().numpy()
        if self.num_classes == 0:
            return output
        else:
            raise NotImplementedError()

    def get_params(self, deep=False) -> dict:
        return{
            "base_estimator": self.base_estimator,
            "num_classes": self.num_classes,
            "input_filter_layer": self.input_filter_layer,
            "n_layers": self.n_layers,
            "n_hidden": self.n_hidden,
            "num_inits": self.num_inits,
            "input_pen": self.input_pen,
            "full_tree_pen": self.full_tree_pen,
            "max_iters": self.max_iters,
            "max_prox_iters": self.max_prox_iters,
            "num_batches": self.num_batches,
            "bootstrap": self.bootstrap,
            "n_jobs": self.n_jobs,
            # "out_model_file": self.out_model_file,
        }

    def set_params(self, **param_dict):
        print("SET PARAMS", param_dict)
        if "base_estimator" in param_dict:
            self.base_estimator = param_dict["base_estimator"]
        if "num_classes" in param_dict:
            self.num_classes = param_dict["num_classes"]
        if "input_filter_layer" in param_dict:
            self.input_filter_layer = param_dict["input_filter_layer"]
        if "n_layers" in param_dict:
            self.n_layers = param_dict["n_layers"]
        if "n_hidden" in param_dict:
            self.n_hidden = param_dict["n_hidden"]
        if "num_inits" in param_dict:
            self.num_inits = param_dict["num_inits"]
        if "input_pen" in param_dict:
            self.input_pen = param_dict["input_pen"]
        if "full_tree_pen" in param_dict:
            self.full_tree_pen = param_dict["full_tree_pen"]
        if "max_iters" in param_dict:
            self.max_iters = param_dict["max_iters"]
        if "max_prox_iters" in param_dict:
            self.max_prox_iters = param_dict["max_prox_iters"]             
        if "num_batches" in param_dict:
            self.num_batches = param_dict["num_batches"]
        if "bootstrap" in param_dict:
            self.bootstrap = param_dict["bootstrap"]     
        if "n_jobs" in param_dict:
            self.n_jobs = param_dict["n_jobs"]      
        # if "out_model_file" in param_dict:
        #     self.out_model_file = param_dict["out_model_file"]

        return self

        #for reference
        #  self.base_estimator=base_estimator
        # self.num_classes=num_classes
        # self.input_filter_layer=input_filter_layer
        # self.n_layers=n_layers
        # self.n_hidden=n_hidden
        # self.num_inits=num_inits
        # self.max_iters=max_iters
        # self.max_prox_iters=max_prox_iters
        # self.bootstrap=bootstrap
        # self.full_tree_pen=full_tree_pen
        # self.input_pen=input_pen
        # self.num_batches=num_batches
        # self.n_jobs=n_jobs   