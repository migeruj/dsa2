# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

python source/run_train.py  run_train --config_name elasticnet  --path_data_train data/input/train/    --path_output data/output/a01_elasticnet/

activate py36 && python source/run_train.py  run_train   --n_sample 100  --config_name lightgbm  --path_model_config source/config_model.py  --path_output /data/output/a01_test/     --path_data_train /data/input/train/

"""
import warnings
warnings.filterwarnings('ignore')
import sys, os, json, importlib

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

DEBUG = True

####################################################################################################
####################################################################################################
from util_feature import   load, save_list, load_function_uri, save
from run_preprocess import  preprocess, preprocess_load


"""
### bug with logger
from util import logger_class
logger = logger_class()

def log(*s):
    logger.log(*s, level=1)

def log2(*s):
    logger.log(*s, level=2)

def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    log(sjump,  df.head(n))
"""


def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)


def log2(*s, n=0, m=0):
    if DEBUG :
        sspace = "#" * n
        sjump = "\n" * m
        ### Implement pseudo Logging
        print(sjump, sspace, s, sspace, flush=True)


def save_features(df, name, path):
    if path is not None :
       os.makedirs( f"{path}/{name}", exist_ok=True)
       df.to_parquet( f"{path}/{name}/features.parquet")


def model_dict_load(model_dict, config_path, config_name, verbose=True):
    """
       load the model dict from the python config file.
    :param model_dict:
    :param config_path:
    :param config_name:
    :param verbose:
    :return:
    """
    if model_dict is None :
       log("#### Model Params Dynamic loading  ###############################################")
       model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
       model_dict     = model_dict_fun()   ### params
    if verbose : log( model_dict )
    return model_dict


####################################################################################################
##### train    #####################################################################################
def map_model(model_name):
    """
      Get the Class of the object stored in source/models/
    :param model_name:   model_sklearn
    :return: model module

    """

    ##### Custom folder
    if ".py" in model_name :
       path = os.path.parent(model_name)
       sys.path.append(path)
       mod = os.path.basename(model_name)
       modelx = importlib.import_module(mod)
       return modelx


    ##### Local folder
    model_file = model_name.split(":")[0]
    if  'optuna' in model_name : model_file = 'optuna_lightgbm'

    try :
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.{model_file}'
       modelx = importlib.import_module(mod)

    except :
        ### All SKLEARN API
        ### ['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod)

    return modelx


def mlflow_register(dfXy, model_dict: dict, stats: dict, mlflow_pars:dict ):
    log("#### Using mlflow #########################################################")
    # def register(run_name, params, metrics, signature, model_class, tracking_uri= "sqlite:///local.db"):
    from run_mlflow import register
    from mlflow.models.signature import infer_signature

    train_signature = dfXy[model_dict['data_pars']['cols_model']]
    y_signature     = dfXy[model_dict['data_pars']['coly']]
    signature       = infer_signature(train_signature, y_signature)

    register( run_name    = model_dict['global_pars']['config_name'],
             params       = model_dict['global_pars'],
             metrics      = stats["metrics_test"],
             signature    = signature,
             model_class  = model_dict['model_pars']["model_class"],
             tracking_uri = mlflow_pars.get( 'tracking_db', "sqlite:///mlflow_local.db")
            )


def train(model_dict, dfX, cols_family, post_process_fun):
    """  Train the model using model_dict, save model, save prediction
    :param model_dict:  dict containing params
    :param dfX:  pd.DataFrame
    :param cols_family: dict of list containing column names
    :param post_process_fun:
    :return: dfXtrain , dfXval  DataFrame containing prediction.
    """
    model_pars, compute_pars = model_dict['model_pars'], model_dict['compute_pars']
    data_pars                = model_dict['data_pars']
    model_name, model_path   = model_pars['model_class'], model_dict['global_pars']['path_train_model']
    metric_list              = compute_pars['metric_list']

    assert  'cols_model_type2' in data_pars, 'Missing cols_model_type2, split of columns by data type '
    log2(data_pars['cols_model_type2'])


    log("#### Model Input preparation #########################################################")
    log(dfX.shape)
    dfX    = dfX.sample(frac=1.0)
    itrain = int(0.6 * len(dfX))
    ival   = int(0.8 * len(dfX))
    colsX  = data_pars['cols_model']
    coly   = data_pars['coly']
    log('Model colsX',colsX)
    log('Model coly', coly)
    log('Model column type: ',data_pars['cols_model_type2'])

    data_pars['data_type'] = 'ram'
    data_pars['train'] = {'Xtrain' : dfX[colsX].iloc[:itrain, :],
                          'ytrain' : dfX[coly].iloc[:itrain],
                          'Xtest'  : dfX[colsX].iloc[itrain:ival, :],
                          'ytest'  : dfX[coly].iloc[itrain:ival],

                          'Xval'   : dfX[colsX].iloc[ival:, :],
                          'yval'   : dfX[coly].iloc[ival:],
                          }


    log("#### Init, Train ############################################################")
    # from config_model import map_model    
    modelx = map_model(model_name)    
    log(modelx)
    modelx.reset()
    modelx.init(model_pars, compute_pars=compute_pars)

    if 'optuna' in model_name:
        modelx.fit(data_pars, compute_pars)
        # No need anymore
        # modelx.model.model_pars['optuna_model'] = modelx.fit(data_pars, compute_pars)
    else:
        modelx.fit(data_pars, compute_pars)


    log("#### Transform ################################################################")
    dfX2 = modelx.transform(dfX[colsX], compute_pars=compute_pars)
    dfX2.index = dfX.index

    for coli in dfX2.columns :
       dfX2[coli]            = dfX2[coli].apply(lambda  x : post_process_fun(x) )

    log("Actual    : ",  dfX[colsX])
    log("Prediction: ",  dfX2)

    log("#### Metrics ###############################################################")
    from util_feature import  metrics_eval
    metrics_test = metrics_eval(metric_list,
                                ytrue       = dfX[coly].iloc[ival:],
                                ypred       = dfX[coly + '_pred'].iloc[ival:],
                                ypred_proba = ypred_proba_val )
    stats = {'metrics_test' : metrics_test}
    log(stats)


    log("### Saving model, dfX, columns #############################################")
    log(model_path + "/model.pkl")
    os.makedirs(model_path, exist_ok=True)
    save(colsX, model_path + "/colsX.pkl")
    save(coly,  model_path + "/coly.pkl")
    modelx.save(model_path, stats)


    log("### Reload model,            ###############################################")
    log(modelx.model.model_pars, modelx.model.compute_pars)
    a = load(model_path + "/model.pkl")
    log("Reload model pars", a.model_pars)
    
    return dfX2.iloc[:ival, :].reset_index(), dfX2.iloc[ival:, :].reset_index(), stats


####################################################################################################
############CLI Command ############################################################################
def run_train(config_name, config_path="source/config_model.py", n_sample=5000,
              mode="run_preprocess", model_dict=None, return_mode='file', **kw):
    """
      Configuration of the model is in config_model.py file
    :param config_name:
    :param config_path:
    :param n_sample:
    :return:
    """
    model_dict  = model_dict_load(model_dict, config_path, config_name, verbose=True)

    m           = model_dict['global_pars']
    path_data_train   = m['path_data_train']
    path_train_X      = m.get('path_train_X', path_data_train + "/features.zip") #.zip
    path_train_y      = m.get('path_train_y', path_data_train + "/target.zip")   #.zip

    path_output         = m['path_train_output']
    # path_model          = m.get('path_model',          path_output + "/model/" )
    path_pipeline       = m.get('path_pipeline',       path_output + "/pipeline/" )
    path_features_store = m.get('path_features_store', path_output + '/features_store/' )  #path_data_train replaced with path_output, because preprocessed files are stored there
    path_check_out      = m.get('path_check_out',      path_output + "/check/" )
    log(path_output)


    log("#### load raw data column family  ###############################################")
    cols_group = model_dict['data_pars']['cols_input_type']  ### Raw
    log(cols_group)


    log("#### Preprocess  ################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']
     
    if mode == "run_preprocess" :
        dfXy, cols      = preprocess(path_train_X, path_train_y,
                                     path_pipeline,    ### path to save preprocessing pipeline
                                     cols_group,       ### dict of column family
                                     n_sample,
                                     preprocess_pars,
                                     path_features_store  ### Store intermediate dataframe
                                     )

    elif mode == "load_preprocess"  :  #### Load existing data
        dfXy, cols      = preprocess_load(path_train_X, path_train_y, path_pipeline, cols_group, n_sample,
                                          preprocess_pars,  path_features_store=path_features_store)


    log("#### Extract column names  #####################################################")
    ### Actual column names for Model Input :  label y and Input X (colnum , colcat)
    model_dict['data_pars']['coly']       = cols['coly']
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])


    #### Col Group by column type : Sparse, continuous, .... (ie Neural Network feed Input
    ## 'coldense' = [ 'colnum' ]     'colsparse' = ['colcat' ]
    model_dict['data_pars']['cols_model_type2'] = {}
    for colg, colg_list in model_dict['data_pars'].get('cols_model_type', {}).items() :
        model_dict['data_pars']['cols_model_type2'][colg] = sum([  cols[colgroup] for colgroup in colg_list ]   , [])


    log("#### Train model: #############################################################")
    log(str(model_dict)[:1000])
    post_process_fun      = model_dict['model_pars']['post_process_fun']
    dfXy, dfXytest,stats  = train(model_dict, dfXy, cols, post_process_fun)


    log("#### Register model ##########################################################")
    mlflow_pars = model_dict.get('compute_pars', {}).get('mlflow_pars', None)
    if mlflow_pars is not None:
        mlflow_register(dfXy, model_dict, stats, mlflow_pars)


    if return_mode == 'dict' :
        return { 'dfXy' : dfXy, 'dfXytest': dfXytest, 'stats' : stats   }

    else :
        log("#### Export ##################################################################")
        os.makedirs(path_check_out, exist_ok=True)
        dfXy.to_parquet(path_check_out + "/dfX.parquet")  # train input data generate parquet
        dfXytest.to_parquet(path_check_out + "/dfXtest.parquet")  # Test input data  generate parquet
        log("######### Finish #############################################################", )



def run_model_check(path_output, scoring):
    """
    :param path_output:
    :param scoring:
    :return:
    """
    import pandas as pd
    try :
        #### Load model
        from source.util_feature import load
        from source.models import model_sklearn as modelx
        import sys
        from source import models
        sys.modules['models'] = models

        dir_model    = path_output
        modelx.model = load( dir_model + "/model/model.pkl" )
        stats        = load( dir_model + "/model/info.pkl" )
        colsX        = load( dir_model + "/model/colsX.pkl"   )
        coly         = load( dir_model + "/model/coly.pkl"   )
        print(stats)
        print(modelx.model.model)

        ### Metrics on test data
        log(stats['metrics_test'])

        #### Loading training data  ######################################################
        dfX     = pd.read_csv(dir_model + "/check/dfX.csv")  #to load csv
        #dfX = pd.read_parquet(dir_model + "/check/dfX.parquet")    #to load parquet
        dfy     = dfX[coly]
        colused = colsX

        dfXtest = pd.read_csv(dir_model + "/check/dfXtest.csv")    #to load csv
        #dfXtest = pd.read_parquet(dir_model + "/check/dfXtest.parquet"    #to load parquet
        dfytest = dfXtest[coly]
        print(dfX.shape,  dfXtest.shape )


        #### Feature importance on training data  #######################################
        from util_feature import  feature_importance_perm
        lgb_featimpt_train,_ = feature_importance_perm(modelx, dfX[colused], dfy,
                                                       colused,
                                                       n_repeats=1,
                                                       scoring=scoring)
        print(lgb_featimpt_train)
    except :
        pass


if __name__ == "__main__":
    import fire
    fire.Fire()




