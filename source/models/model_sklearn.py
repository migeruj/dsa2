# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Template for sklearn type of model:

    #### 1. Global variables    : global variables **model** and **session**.

    #### 2.  init method        : init method to initialize global variables `model and session`

    #### 3. class Model         :   storing model details and parameters.

    #### 4. preprocess method   :   preprocessing dataset.

    #### 5. fit method          :   fitting the defined model and inputted data.

    #### 6. predict method      :   predicting using the fitted model.

    #### 7. save method         :   saving the model in the pickle file.

    #### 8. load_model method   :   loading, the model saved in a pickle file.

    #### 9. load_info method    :   loading the in mation stored in the pickle file.

    #### 10. get_dataset method :   retrieving the dataset.

    #### 11. get_params method  :   retrieving parameters.



"""
import os
import pandas as pd, numpy as np, scipy as sci

import sklearn
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier


try :
   from supervised.automl import *
except:
    print('cannot import automl')    

try :
    #### All are Un-supervised Model
    from pyod.models.abod  import *
    from pyod.models.auto_encoder import *
    from pyod.models.cblof import *
    from pyod.models.cof import *
    from pyod.models.combination import *
    from pyod.models.copod import *
    from pyod.models.feature_bagging import *
    from pyod.models.hbos import *
    from pyod.models.iforest import *
    from pyod.models.knn import *
    from pyod.models.lmdd import *
    from pyod.models.loda import *
    from pyod.models.lof import *
    from pyod.models.loci import *
    from pyod.models.lscp import *
    from pyod.models.mad import *
    from pyod.models.mcd import *
    from pyod.models.mo_gaal import *
    from pyod.models.ocsvm import *
    from pyod.models.pca import *
    from pyod.models.sod import *
    from pyod.models.so_gaal import *
    from pyod.models.sos import *
    from pyod.models.vae import *
    from pyod.models.xgbod import *
except :
    print("cannot import pyod")
 
####################################################################################################
VERBOSE = True


# MODEL_URI = get_model_uri(__file__)

def log(*s):
    print(*s, flush=True)


def log3(*s):
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
            model_class = globals()[model_pars['model_class']]
            self.model = model_class(**model_pars['model_pars'])
            if VERBOSE: log(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")
    if VERBOSE: log(Xtrain.shape, model.model)

    if "LGBM" in model.model_pars['model_class']:
        model.model.fit(Xtrain, ytrain, eval_set=[(Xtest, ytest)], **compute_pars.get("compute_pars", {}))
    else:
        model.model.fit(Xtrain, ytrain, **compute_pars.get("compute_pars", {}))


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    data_pars['train'] = True
    Xval, yval = get_dataset(data_pars, task_type="eval")
    # ypred      = model.model.predict(Xval)
    ypred = predict(Xval, data_pars, compute_pars, out_pars)

    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {'metric_name': 'mae'})

    scorer = {
        "rmse": sklearn.metrics.mean_squared_error,
        "mae": sklearn.metrics.mean_absolute_error
    }[mpars['metric_name']]

    mpars2 = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred, **mpars2)

    ddict = [{"metric_val": score_val, 'metric_name': mpars['metric_name']}]

    return ddict


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y

    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    ypred = model.model.predict(Xpred)
    #ypred = post_process_fun(ypred)
    
    ypred_proba = None  ### No proba    
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred) 
    return ypred, ypred_proba


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


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values,
                                                        stratify=dfy.values,test_size=0.1)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  : 
      "file" :
    """
    # log(data_pars)
    data_type  = data_pars.get('type', 'ram')
    cols_type  = data_pars.get('cols_model_type2', {})   #### Split input by Sparse, Continous
    cols_model = data_pars['cols_model']
    coly       = data_pars['coly']

    log3("Cols Type:", cols_type)

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


def get_params_sklearn(deep=False):
    return model.model.get_params(deep=deep)


def get_params(param_pars={}, **kw):
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")
