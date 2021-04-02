# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""



"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
try   : verbosity = int(json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../../config.json", mode='r'))['verbosity'])
except Exception as e : verbosity = 2
#raise Exception(f"{e}")

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)

def os_makedirs(dir_or_file):
    if os.path.isfile(dir_or_file) :os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else : os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)

####################################################################################################
global model, session
def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None

def reset():
    global model, session
    model, session = None, None


########Custom Model #########################################################################################
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier


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


def model_automl():
    try :
       from supervised.automl import AutoML
    except:
       os.system('pip install mljar-supervised==0.10.2  lightgbm==3.0.0 catboost==0.24.4 joblib==1.0.1 cloudpickle==1.3.0 pyarrow>=2.0.0 tabulate==0.8.7 matplotlib>=3.2.2 dtreeviz==1.0 shap seaborn==0.10.1 wordcloud==1.8.1 category_encoders==2.2.2 optuna==2.6.0   --no-deps')
       from supervised.automl import AutoML
    model_class = AutoML
    return model_class


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None
        , compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            if  model_pars['model_class'] == 'AutoML':
                model_class = model_automl()
            else :
                model_class = globals()[model_pars['model_class']]

            self.model = model_class(**model_pars['model_pars'])
            log(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")
    log(Xtrain.shape, model.model)

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

    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    ypred = model.model.predict(Xpred)
    #ypred = post_process_fun(ypred)
    
    ypred_proba = None  ### No proba    
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred) 
    return ypred, ypred_proba


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




#################################################################################################################
def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim=11
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(
              n_samples=10000, n_features=ndim, n_classes=1, n_redundant = 0, n_informative=ndim )
    df = pd.DataFrame(X,  columns= colnum)
    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))
    df[coly]   = y.reshape(-1, 1)
    # log(df)
    return df, colnum, colcat, coly


def train_test_split2(df, coly):
    log3(df.dtypes)
    y = df[coly] ### If clonassificati
    X = df.drop(coly,  axis=1)
    log3('y', np.sum(y[y==1]) , X.head(3))
    ######### Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)

    #####
    # y = y.astype('uint8')
    num_classes                                = len(set(y_train_full.values.ravel()))

    return X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes


def test(n_sample          = 1000):
    df, colnum, colcat, coly = test_dataset_classi_fake(nrows= n_sample)
    X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes  = train_test_split2(df, coly)
  
    cols_input_type_1 = []
    #### Matching Big dict  ##################################################
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)


    m = {
    'model_pars': {
        'model_class':  "model_sklearn.py::LightGBM"
        ,'model_pars' : {  }  
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
        'cols_model_group': [ 'colnum_bin',   'colcat_bin',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                            }
        ###################################################  
        ,'train': {'Xtrain': X_train, 'ytrain': y_train,
                   'Xtest': X_valid,  'ytest':  y_valid},
                'eval': {'X': X_valid,  'y': y_valid},
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
    """https://github.com/manujosephv/pytorch_tabular/blob/main/tests/test_mdn.py
    

    """
    ll = [
        ('torch_tabular.py::CategoryEmbeddingModelConfig', 
            {   'task': "classification",
                'metrics' : ["f1","accuracy"],
                'metrics_params' : [{"num_classes":num_classes},{}]
            }  
        ),
    ]
    for cfg in ll:
        log("******************************************** New Model ********************************************")
        log(f"******************************************** {cfg[0]} ********************************************")
        reset()
        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg[0]
        m['model_pars']['model_pars']  = {**m['model_pars']['model_pars'] , **cfg[1] }

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

        log('Model architecture:')
        log(model.model)
        reset()



def test2():
    """
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

    https://pyod.readthedocs.io/en/latest/pyod.html

    :return:
    """
    global model
    try:
        from pmlb import fetch_data, classification_dataset_names
    except:
        log("Installing pmlb...")
        os.system("pip install pmlb")
        log("Pmlb Installed")
        from pmlb import fetch_data, classification_dataset_names


    ####
    m = template_dict()
    ll= [ 'PCA', 'SO_GAAL', 'VAE', 'HBOS' ]

    for dataset in classification_dataset_names :
      for modeli in  ll:
         m['model_class'] = modeli








if __name__ == "__main__":
    import fire
    fire.Fire()


