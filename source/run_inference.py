# -*- coding: utf-8 -*-
"""
python source/run_inference.py  run_predict  --n_sample 1000  --config_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data_train /data/input/train/

"""
import warnings,sys, json, gc, os, pandas as pd, importlib
warnings.filterwarnings('ignore')

#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

####################################################################################################
try   : verbosity = int(json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../config.json", mode='r'))['verbosity'])
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
#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
log2(root)


from util_feature import load, load_function_uri, load_dataset

def model_dict_load(model_dict, config_path, config_name, verbose=True):
    """model_dict_load
    Args:
        model_dict ([type]): [description]
        config_path ([type]): [description]
        config_name ([type]): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if model_dict is None :
       log("#### Model Params Dynamic loading  ###############################################")
       model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
       model_dict     = model_dict_fun()   ### params
    if verbose : log( model_dict )
    return model_dict


def map_model(model_name):
    """ Get the Class of the object stored in source/models/
    :param model_name:   model_sklearn
    :return: model module
    """

    ##### Custom folder
    if ".py" in model_name :
       ### Asbolute path of the file
       path = os.path.dirname(os.path.abspath(model_name))
       sys.path.append(path)
       mod    = os.path.basename(model_name).replace(".py", "")
       modelx = importlib.import_module(mod)
       return modelx

    ##### Repo folder
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


def predict(model_name, path_model, dfX, cols_family, model_dict):
    """
    Arguments:
        model_name {[str]} -- [description]
        path_model {[str]} -- [description]
        dfX {[DataFrame]} -- [description]
        cols_family {[dict]} -- [description]

    Returns: ypred
        [numpy.array] -- [vector of prediction]
    """
    log("#### Load  model class  ############################################")
    modelx = map_model(model_name)
    assert modelx is not None, "cannot load modelx, " + path_model
    modelx.reset()
    log2(modelx, path_model)
    sys.path.append( root)    #### Needed due to import source error

    log("#### Load existing model weights  #################################")
    log2(path_model + "/model/")
    # modelx.model = load(path_model + "/model//model.pkl")
    # modelx.model = load(path_model + "/model.pkl")
    modelx.load_model( path_model)
    colsX       = load(path_model + "/colsX.pkl")   ## column name

    assert colsX is not None, "cannot load colsx, " + path_model
    assert modelx.model is not None, "cannot load modelx, " + path_model
    log2("#### modelx\n", modelx.model)

    log("### Prediction  ###################################################")
    dfX  = dfX.reindex(columns=colsX)   #reindex included
    ypred_tuple = modelx.predict(dfX, data_pars    = model_dict['data_pars'],
                                      compute_pars = model_dict['compute_pars']                           )
    log2('ypred shape', str(ypred_tuple)[:100] )
    return ypred_tuple


####################################################################################################
############CLI Command ############################################################################
def run_predict(config_name, config_path, n_sample=-1,
                path_data=None, path_output=None, pars={}, model_dict=None):

    log("#### Run predict  ###############################################################")
    model_dict = model_dict_load(model_dict, config_path, config_name, verbose=True)
    model_class      = model_dict['model_pars']['model_class']

    m                = model_dict['global_pars']
    path_data        = m['path_pred_data']   if path_data   is None else path_data
    path_pipeline    = m['path_pred_pipeline']    #   path_output + "/pipeline/" )
    path_model       = m['path_pred_model']
    path_output      = m['path_pred_output'] if path_output is None else path_output
    log(path_data, path_model, path_output)

    pars = {'cols_group': model_dict['data_pars']['cols_input_type'],
            'pipe_list' : model_dict['model_pars']['pre_process_pars']['pipe_list']}


    log("#### Run preprocess  ###########################################################")
    from run_preprocess import preprocess_inference   as preprocess
    colid            = load(f'{path_pipeline}/colid.pkl')
    df               = load_dataset(path_data, path_data_y=None, colid=colid, n_sample=n_sample)
    dfX, cols        = preprocess(df, path_pipeline, preprocess_pars=pars)
    coly = cols["coly"]  


    log("#### Extract column names  #########################################################")
    ### Actual column names for Model Input :  label y and Input X (colnum , colcat), remove duplicate names
    ###  [  'colcat', 'colnum'
    model_dict['data_pars']['coly']       = cols['coly']
    model_dict['data_pars']['cols_model'] = list(set(sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , []) ))


    #### Flatten Col Group by column type : Sparse, continuous, .... (ie Neural Network feed Input, remove duplicate names
    ## 'coldense' = [ 'colnum' ]     'colsparse' = ['colcat' ]
    model_dict['data_pars']['cols_model_type2'] = {}
    for colg, colg_list in model_dict['data_pars'].get('cols_model_type', {}).items() :
        model_dict['data_pars']['cols_model_type2'][colg] = list(set(sum([  cols[colgroup] for colgroup in colg_list ]   , [])))


    log("############ Prediction  ##########################################################" )
    ypred, yproba    = predict(model_class, path_model, dfX, cols, model_dict)

    post_process_fun        = model_dict['model_pars']['post_process_fun']
    df[ coly + "_pred"]     = ypred
    df[ coly + "_pred"]     = df[coly + '_pred'].apply(lambda  x : post_process_fun(x) )
    if yproba is not None :
       df[ coly + "_pred_proba"] = yproba


    log("############ Saving prediction  ###################################################" )
    log(ypred.shape, path_output)
    os.makedirs(path_output, exist_ok=True)
    df.to_csv(f"{path_output}/prediction.csv")
    log(df.head(8))


    log("###########  Export Specific ######################################################")
    df[cols["coly"]] = ypred
    df[[cols["coly"]]].to_csv(f"{path_output}/pred_only.csv")



##############################################################################################
def run_data_check(path_data, path_data_ref, path_model, path_output, sample_ratio=0.5):
    """
     Calcualata Dataset Shift before prediction.
    """
    from run_preprocess import preprocess_inference   as preprocess
    path_output   = root + path_output
    path_data     = root + path_data
    path_data_ref = root + path_data_ref
    path_pipeline = root + path_model + "/pipeline/"

    os.makedirs(path_output, exist_ok=True)
    colid          = load(f'{path_pipeline}/colid.pkl')

    df1                = load_dataset(path_data_ref,colid=colid)
    dfX1, cols_family1 = preprocess(df1, path_pipeline)

    df2                = load_dataset(path_data,colid=colid)
    dfX2, cols_family2 = preprocess(df2, path_pipeline)

    colsX       = cols_family1["colnum_bin"] + cols_family1["colcat_bin"]
    dfX1        = dfX1[colsX]
    dfX2        = dfX2[colsX]

    from util_feature import pd_stat_dataset_shift
    nsample     = int(min(len(dfX1), len(dfX2)) * sample_ratio)
    metrics_psi = pd_stat_dataset_shift(dfX2, dfX1,
                                        colsX, nsample=nsample, buckets=7, axis=0)
    metrics_psi.to_csv(f"{path_output}/prediction_features_metrics.csv")
    log(metrics_psi)



###########################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()