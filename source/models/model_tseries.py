# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Template for tseries type of model:


"""
import os, pandas as pd, numpy as np, scipy as sci, sklearn
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformers.single_series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
)
from sktime.utils.plotting import plot_series
from sktime.forecasting.compose import (
    TransformedTargetForecaster,
    ReducedRegressionForecaster
)

try :
   from supervised.automl import *
except:
    print('cannot import automl')

####################################################################################################
VERBOSE = True

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

def reset():
    global model, session
    model, session = None, None

####################################################################################################
class myModel(object):
    pass




####################################################################################################
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



def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session

    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    Xpred_fh = ForecastingHorizon(Xpred.index, is_relative=False)

    ypred = model.model.predict(Xpred_fh)

    ypred_proba = None  ### No proba
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




####################################################################################################################
def test_dataset_tseries()
   pass


def LighGBM_forecaster(lightgbm_pars= {'objective':'quantile', 'alpha': 0.5},
                       forecaster_pars = {'window_length': 4, 'strategy' : "recursive" }):
    """
    """
    #Initialize Light GBM Regressor
    regressor = lgb.LGBMRegressor(**lightgbm_params)

    #1.Separate the Seasonal Component.
    #2.Fit a forecaster for the trend.
    #3.Fit a Autoregressor to the resdiual(autoregressing on four historic values).
    forecaster = ReducedRegressionForecaster(
                    regressor=regressor, **forecaster_pars  #hyper-paramter to set recursive strategy
                    )
    return forecaster


def test0(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    global model, session
    df, coly, coldate, colcat = test_dataset_tseries()

    #### Matching Big dict  ##################################################
    df = df.set_index(coldate)  #### Date as
    X  = df.drop(coly)
    y  = df[coly]

    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    sktime_y_train，sktime_y_test = temporal_train_test_split(y, test_size=0.2)

    #A 10 percent and 90 percent prediction interval(0.1,0.9 respectively).
    quantiles = [.1, .5, .9] #Hyper-parameter "alpha" in Light GBM
    #Capture forecasts for 10th/median/90th quantile, respectively.
    forecasts = []

    #Iterate for each quantile.
    for alpha in quantiles:
        forecaster = LighGBM_forecaster(lightgbm_pars= {'objective':'quantile', 'alpha': 0.5} )

        #Fit on Training data.
        forecaster.fit(y_train)

        #Forecast the values.
        #Initialize ForecastingHorizon class to specify the horizon of forecast
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        y_pred = forecaster.predict(fh)


        #List of forecasts made for each quantile.
        y_pred.index.name="date"
        y_pred.name=f"predicted_sales_q_{alpha}"
        forecasts.append(y_pred)

    #Append the actual data for plotting.
    store1_agg_monthly.index.name = "date"
    store1_agg_monthly.name = "original"
    forecasts.append(store1_agg_monthly)


    log('Predict data..')
    log(f'Top 5 y_pred: {forecasts[:5]}')








def test2(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    global model, session
    df, colnum, colcat, coly = test_dataset()

    #### Matching Big dict  ##################################################
    X = df
    y = df[coly].astype('uint8')
    log('y', np.sum(y[y==1]) )

    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    sktime_y_train，sktime_y_test = temporal_train_test_split(y, test_size=0.2)

    def get_transformed_target_forecaster(alpha,params):

        #Initialize Light GBM Regressor

        regressor = lgb.LGBMRegressor(alpha = alpha,**params)
    #-----------------------Forecaster Pipeline-----------------

        #1.Separate the Seasonal Component.
        #2.Fit a forecaster for the trend.
        #3.Fit a Autoregressor to the resdiual(autoregressing on four historic values).

        forecaster = ReducedRegressionForecaster(
                        regressor=regressor, window_length=4, strategy="recursive" #hyper-paramter to set recursive strategy
                        )

        return forecaster

    params = {
        'objective':'quantile'
    }
    #A 10 percent and 90 percent prediction interval(0.1,0.9 respectively).
    quantiles = [.1, .5, .9] #Hyper-parameter "alpha" in Light GBM
    #Capture forecasts for 10th/median/90th quantile, respectively.
    forecasts = []
    #Iterate for each quantile.
    for alpha in quantiles:

        forecaster = get_transformed_target_forecaster(alpha,params)

        #Initialize ForecastingHorizon class to specify the horizon of forecast
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        #Fit on Training data.
        forecaster.fit(y_train)

        #Forecast the values.
        y_pred = forecaster.predict(fh)

        #List of forecasts made for each quantile.
        y_pred.index.name="date"
        y_pred.name=f"predicted_sales_q_{alpha}"
        forecasts.append(y_pred)

    #Append the actual data for plotting.
    store1_agg_monthly.index.name = "date"
    store1_agg_monthly.name = "original"
    forecasts.append(store1_agg_monthly)


    log('Predict data..')
    log(f'Top 5 y_pred: {forecasts[:5]}')
    reset()


    num_classes = len(set(y_train_full[coly].values.ravel()))
    log(X_train)


    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return int(y)

    def pre_process_fun(y):
        return int(y)


    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  "torch_tabular.py::CategoryEmbeddingModelConfig"

        # Type of target prediction, evaluation metrics
        ,'model_pars' : {
                        # 'task': "classification",
                        # 'metrics' : ["f1","accuracy"],
                        # 'metrics_params' : [{"num_classes":num_classes},{}]
                        }

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },

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
    """

    """
    ll = [
        ('model_tseries.py::LightGBMregressor',
            {   'task': "classification",
                'metrics' : ["f1","accuracy"],
                'metrics_params' : [{"num_classes":num_classes},{}]
            }
        ),

    ]
    for cfg in ll:
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



if __name__ == "__main__":
    import fire
    fire.Fire()
    # test()
