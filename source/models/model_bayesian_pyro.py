# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

"""
import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import sklearn

from torch import nn
from pyro.nn import PyroModule
import logging

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroSample

####################################################################################################
VERBOSE = False

def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session


def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            ###############################################################
            class BayesianRegression(PyroModule):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.linear = PyroModule[nn.Linear](in_features, out_features)
                    self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
                    self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

                def forward(self, x, y=None):
                    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
                    mean = self.linear(x).squeeze(-1)
                    with pyro.plate("data", x.shape[0]):
                        obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
                    return mean

            input_width = model_pars['model_pars']['input_width']
            y_width = model_pars['model_pars'].get('y_width', 1)
            self.model = BayesianRegression(input_width, y_width)
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")

    Xtrain = torch.tensor(Xtrain.values, dtype=torch.float)
    Xtest = torch.tensor(Xtest.values, dtype=torch.float)
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    ytest = torch.tensor(ytest.values, dtype=torch.float)

    if VERBOSE: log(Xtrain, model.model)

    ###############################################################
    compute_pars2 = compute_pars.get('compute_pars', {})
    n_iter = compute_pars2.get('n_iter', 1000)
    lr = compute_pars2.get('learning_rate', 0.01)
    method = compute_pars2.get('method', 'svi_elbo')

    guide = AutoDiagonalNormal(model.model)
    adam = pyro.optim.Adam({"lr": lr})

    ### SVI + Elbo is faster than HMC
    svi = SVI(model.model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for j in range(n_iter):
        # calculate the loss and take a gradient step
        loss = svi.step(Xtrain, ytrain)
        losses.append({'loss': loss, 'iteration': j})
        if j % 100 == 0:
            log("[iteration %04d] loss: %.4f" % (j + 1, loss / len(Xtrain)))

    model.guide = guide

    df_loss = pd.DataFrame(losses)
    df_loss['loss'].plot()
    return df_loss


def predict(Xpred=None, data_pars={}, compute_pars=None, out_pars={}, **kw):
    global model, session
    # data_pars['train'] = False

    compute_pars2 = model.compute_pars if compute_pars is None else compute_pars
    num_samples = compute_pars2.get('num_samples', 300)

    ###### Data load
    if Xpred is None:
        Xpred = get_dataset(data_pars, task_type="predict")
    cols_Xpred = list(Xpred.columns)

    max_size = compute_pars2.get('max_size', len(Xpred))

    Xpred = Xpred.iloc[:max_size, :]
    Xpred_ = torch.tensor(Xpred.values, dtype=torch.float)

    ###### Post processing normalization
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y

    from pyro.infer import Predictive
    def summary(samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                # "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                # "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            }
        return site_stats

    predictive = Predictive(model.model, guide=model.guide, num_samples=num_samples,
                            return_sites=("linear.weight", "obs", "_RETURN"))
    pred_samples = predictive(Xpred_)
    pred_summary = summary(pred_samples)

    mu = pred_summary["_RETURN"]
    y = pred_summary["obs"]
    dd = {
        "mu_mean": post_process_fun(mu["mean"].detach().numpy()),
        # "mu_perc_5"    : post_process_fun( mu["5%"].detach().numpy() ),
        # "mu_perc_95"   : post_process_fun( mu["95%"].detach().numpy() ),
        "y_mean": post_process_fun(y["mean"].detach().numpy()),
        # "y_perc_5"     : post_process_fun( y["5%"].detach().numpy() ),
        # "y_perc_95"    : post_process_fun( y["95%"].detach().numpy() ),
        # "true_salary" : y_data,
    }
    for i, col in enumerate(cols_Xpred):
        dd[col] = Xpred[col].values  # "major_PHYSICS": x_data[:, -8],
    # print(dd)
    ypred_mean = pd.DataFrame(dd)
    model.pred_summary = {'pred_mean': ypred_mean, 'pred_summary': pred_summary, 'pred_samples': pred_samples}
    print('stored in model.pred_summary')
    # print(  dd['y_mean'], dd['y_mean'].shape )
    # import pdb; pdb.set_trace()
    return dd['y_mean']


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model0.model
    model.model_pars = model0.model_pars
    model.compute_pars = model0.compute_pars
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd

def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  : 
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')




def y_norm(y, inverse=True, mode='boxcox'):
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 0.6145279599674994  # Optimal boxCox lambda for y
        if inverse:
                y2 = y * width0
                y2 = ((y2 * k1) + 1) ** (1 / k1)
                return y2
        else:
                y1 = (y ** k1 - 1) / k1
                y1 = y1 / width0
                return y1

    if mode == 'norm':
        m0, width0 = 0.0, 350.0  ## Min, Max
        if inverse:
                y1 = (y * width0 + m0)
                return y1

        else:
                y2 = (y - m0) / width0
                return y2
    else:
            return y



def test(nrows=1000):
    """
        nrows : take first nrows from dataset
    """

    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",  
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
        "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  ]
    
    # Target column
    coly        = ["Covertype"]

    log("start")
    global model, session

    root = os.path.join(os.getcwd() ,"ztmp")


    BASE_DIR = Path.home().joinpath( root, 'data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    
    # Download the dataset in case it's missing
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    # Read nrows of only the given columns 
    feature_columns = colnum + colcat + coly
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)


    #### Matching Big dict  ##################################################
    X = df

    #### Regression PLEASE RANDOM VALUES AS TEST
    y = np.random.random(0,1, len(df))  
    log('y', y)

    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)
    num_classes = len(set(y_train_full[coly].values.ravel()))
    log(X_train)


    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')


    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  ""
        
        # Type of target prediction, evaluation metrics
        ,'model_pars' : {'input_width': 112, } 

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },

        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },

        #### catcol INTO integer,   colcat into OneHot
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },

        ],
            }
        },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                    },

    'data_pars': { 'n_sample' : n_sample,

        'download_pars' : None,

        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum',
                              'colcat_binto_onehot',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat_binto_onehot,
                                        'coly' : coly
                                      }

        ###################################################  
        ,'train': {'Xtrain': X_train,
                    'ytrain': y_train,
                        'Xtest': X_valid,
                        'ytest': y_valid},
                'eval': {'X': X_valid,
                        'y': y_valid},
                'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        
        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' : colcat, 
        },
        }
    }

    ##### Running loop
    ll = [
        'model_bayesian_pyro.py::CategoryEmbeddingModelConfig',
    ]
    for cfg in ll:

        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')


        log('Saving model..')
        save(path= "ztmp/data/output/torch_tabular")
        #  os.path.join(root, 'data\\output\\torch_tabular\\model'))

        log('Load model..')
        model, session = load_model(path="ztmp/data/output/torch_tabular")
            
        log('Model architecture:')
        log(model.model)

        log('Model config:')
        log(model.model.config._config_name)
        reset()



if __name__ == "__main__":
    # import fire
    # fire.Fire()
    test()


