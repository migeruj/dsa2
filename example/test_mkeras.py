# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  python mkeras.py  train    > zlog/log_titanic_train.txt 2>&1
  python mkeras.py  predict  > zlog/log_titanic_predict.txt 2>&1


"""
import warnings, copy, os, sys
warnings.filterwarnings("ignore")

####################################################################################
###### Path ########################################################################
root_repo      =  os.path.abspath(os.getcwd()).replace("\\", "/") + "/"     ; print(root_repo)
THIS_FILEPATH  =  os.path.abspath(__file__) 

sys.path.append(root_repo)
from source.util_feature import save,os_get_function_name




def global_pars_update(model_dict,  data_name, config_name):
    print("config_name", config_name)
    dir_data  = root_repo + "/data/"  ; print("dir_data", dir_data)

    m                      = {}
    m["config_path"]       = THIS_FILEPATH  
    m["config_name"]       = config_name

    #### peoprocess input path
    m["path_data_preprocess"] = dir_data + f"/input/{data_name}/train/"

    #### train input path
    dir_data_url              = "https://github.com/arita37/dsa2_data/tree/master/"  #### Remote Data directory
    m["path_data_train"]      = dir_data_url + f"/input/{data_name}/train/"
    m["path_data_test"]       = dir_data_url + f"/input/{data_name}/test/"
    #m["path_data_val"]       = dir_data + f"/input/{data_name}/test/"

    #### train output path
    m["path_train_output"]    = dir_data + f"/output/{data_name}/{config_name}/"
    m["path_train_model"]     = dir_data + f"/output/{data_name}/{config_name}/model/"
    m["path_features_store"]  = dir_data + f"/output/{data_name}/{config_name}/features_store/"
    m["path_pipeline"]        = dir_data + f"/output/{data_name}/{config_name}/pipeline/"


    #### predict  input path
    m["path_pred_data"]       = dir_data + f"/input/{data_name}/test/"
    m["path_pred_pipeline"]   = dir_data + f"/output/{data_name}/{config_name}/pipeline/"
    m["path_pred_model"]      = dir_data + f"/output/{data_name}/{config_name}/model/"

    #### predict  output path
    m["path_pred_output"]     = dir_data + f"/output/{data_name}/pred_{config_name}/"

    #####  Generic
    m["n_sample"]             = model_dict["data_pars"].get("n_sample", 5000)

    model_dict[ "global_pars"] = m
    return model_dict



####################################################################################
##### Params########################################################################
config_default   = "config1"    ### name of function which contains data configuration


# data_name    = "titanic"     ### in data/input/
cols_input_type_1 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   ["Sex", "Embarked" ]
    ,"colnum" :   ["Pclass", "Age","SibSp", "Parch","Fare"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : [  ]
}


####################################################################################
def config1() :
    """
       ONE SINGLE DICT Contains all needed informations for
       used for titanic classification task
    """
    data_name    = "titanic"         ### in data/input/

    # model_class  = "source/models/model_sklearn.py::LightGBM"  ### ACTUAL Class name for

    model_class  = "source/models/keras_widedeep.py"  ### ACTUAL Class name for

    
    n_sample     = 1000

    def post_process_fun(y):   ### After prediction is done
        return  int(y)

    def pre_process_fun(y):    ### Before the prediction is done
        return  int(y)


    model_dict = {"model_pars": {
        ### LightGBM API model   #######################################
         "model_class": model_class
        ,"model_pars" : {


                      'model_name': name,
                      'linear_feat_col'    : linear_feat_col,
                      'dnn_feat_col'       : dnn_feat_col,
                      'behavior_feat_list' : behavior_feat_list,
                      'region_feat_col'    : region_feat_col,
                      'base_feat_col'      : base_feat_col,
                      'task'                  : task,


                      
                      'model_pars': {'optimizer': opt,
                                     'loss': loss,
                                     'metrics': metrics}
                     },



















                        }

        , "post_process_fun" : post_process_fun                    ### After prediction  ##########################################
        , "pre_process_pars" : {"y_norm_fun" :  pre_process_fun ,  ### Before training  ##########################


        ### Pipeline for data processing ##############################
        "pipe_list": [
        #### coly target prorcessing
        {"uri": "source/prepro.py::pd_coly",                 "pars": {}, "cols_family": "coly",       "cols_out": "coly",           "type": "coly"         },


        {"uri": "source/prepro.py::pd_colnum_bin",           "pars": {}, "cols_family": "colnum",     "cols_out": "colnum_bin",     "type": ""             },

        #### catcol INTO integer
        {"uri": "source/prepro.py::pd_colcat_bin",           "pars": {}, "cols_family": "colcat",     "cols_out": "colcat_bin",     "type": ""             },


        ],
               }
        },

      "compute_pars": { "metric_list": ["accuracy_score","average_precision_score"]

                        ,"mlflow_pars" : {}   ### Not empty --> use mlflow
                      },

      "data_pars": { "n_sample" : n_sample,

          "download_pars" : None,

          ### family of columns for raw input data  #########################################################
          "cols_input_type" : cols_input_type_1,


          ### family of columns used for model input  #########################################################
          "cols_model_group": [ "colnum",       ### numerical continuous   
                                "colcat_bin",   ###  category


                              ]

          ### Filter data rows   ##################################################################
         ,"filter_pars": { "ymax" : 2 ,"ymin" : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict




###################################################################################
########## Preprocess #############################################################
### def preprocess(config="", nsample=1000):
from core_run import preprocess

"""
def preprocess(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict["global_pars"]
    print(mdict)

    from source import run_preprocess
    run_preprocess.run_preprocess(config_name   =  config_name,
                                  config_path   =  m["config_path"],
                                  n_sample      =  nsample if nsample is not None else m["n_sample"],

                                  ### Optonal
                                  mode          =  "run_preprocess")
"""



##################################################################################
########## Train #################################################################
from core_run import train
"""
def train(config=None, nsample=None):

    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict["global_pars"]
    print(mdict)
    
    from source import run_train
    run_train.run_train(config_name       =  config_name,
                        config_path       =  m["config_path"],
                        n_sample          =  nsample if nsample is not None else m["n_sample"]
                        )
"""




####################################################################################
####### Inference ##################################################################
# predict(config="", nsample=10000)
from core_run import predict

"""
def predict(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict["global_pars"]


    from source import run_inference
    run_inference.run_predict(config_name = config_name,
                              config_path = m["config_path"],
                              n_sample    = nsample if nsample is not None else m["n_sample"],

                              #### Optional
                              path_data   = m["path_pred_data"],
                              path_output = m["path_pred_output"],
                              model_dict  = None
                              )
"""


###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
    d = { "train" : train, "predict" : predict, "config" : config_default }
    import fire
    fire.Fire(d)
    



