# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

ipython source/models/ztemplate.py  test  --pdb



"""
import os, pandas as pd, numpy as np, sklearn, copy
from sklearn.model_selection import train_test_split

####################################################################################################
verbosity =2

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 :
      print(*s, flush=True)


####################################################################################################
#### Your model input
cols_ref_formodel = ['cols_group_1_name' ]



class Modelcustom(object):

    def __init__(n_wide_cross, n_wide,n_deep, n_feat=8, m_EMBEDDING=10, loss='mse', metric = 'mean_squared_error'):
        """
            YOUR MODEL definition

        """
        pass




























####################################################################################################
####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


class Model(object):
    """
           Generic Wrapper Class for Modelcustom
           Actual model is in Model.model

    """
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
        else:
            log2("data_pars", data_pars)
            model_class = model_pars['model_class']  #
            mdict       = model_pars['model_pars']


            ######### Size the model based on data size  ##############
            mdict['model_pars']['n_columns']  = data_pars['n_columns']





            ######### Create Model Instance  #########################
            self.model  = Modelcustom(**mdict)
            log2(self.model)



def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """ Train the model.model
    """
    global model, session

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle
    
    model.model.fit(Xtrain, ytrain, **cpars)





def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")
    else :
        cols_type   = data_pars['cols_model_type2'] 
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    ypred = model.model.predict(Xpred_tuple )

    ###### Add Probability 
    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)

    #####  Return prediction
    return ypred, ypred_proba






def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

#
def load_model(path=""):
    global model, session
    import dill as pickle





####################################################################################################
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :  ## No Tuple
        return Xtrain

    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )


    return Xtuple_train


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes to feed Model
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel   ### Column GROUP defined for the model

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### Defined source/run_train.py ---> data_pars ---> get_dataset
        ### ##3 Sparse, Continuous
        cols_type_received     = data_pars.get('cols_model_type2', {} )  

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)


            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



####################################################################################################
############ Do not change #########################################################################
def test(config=''):
    """
        Group of columns for the input model
           cols_input_group = [ ]
          for cols in cols_input_group,

    :param config:
    :return:
    """

    X = pd.DataFrame( np.random.rand(100,30), columns= [ 'col_' +str(i) for i in range(30)] )
    y = pd.DataFrame( np.random.binomial(n=1, p=0.5, size=[100]), columns = ['coly'] )
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    log(X_train.shape, )
    ##############################################################
    ##### Generate column actual names from
    colnum = [ 'col_0', 'col_11', 'col_8']
    colcat = [ 'col_13', 'col_17', 'col_13', 'col_9']

    cols_input_type_1 = {
        'colnum' : colnum,
        'colcat' : colcat
    }

    ###### Keras has 1 tuple input    ###########################
    colg_input = {
      'cols_cross_input':  ['colnum', 'colcat' ],
      'cols_deep_input':   ['colnum', 'colcat' ],
    }

    cols_model_type2= {}
    for colg, colist in colg_input.items() :
        cols_model_type2[colg] = []
        for colg_i in colist :
          cols_model_type2[colg].extend( cols_input_type_1[colg_i] )


    ##################################################################################
    model_pars = {'model_class': 'WideAndDeep',
                  'model_pars': {},
                }
    
    n_sample = 100
    data_pars = {'n_sample': n_sample,
                  'cols_input_type': cols_input_type_1,

                  'cols_model_group': ['colnum',
                                       'colcat',
                                       # 'colcross_pair'
                                       ],

                  'cols_model_type2' : cols_model_type2


        ### Filter data rows   #######################3############################
        , 'filter_pars': {'ymax': 2, 'ymin': -1}
                  }

    data_pars['train'] ={'Xtrain': X_train,  'ytrain': y_train,
                         'Xtest': X_test,  'ytest': y_test}
    data_pars['eval'] =  {'X': X_test,
                          'y': y_test}
    data_pars['predict'] = {'X': X_test}

    compute_pars = { 'compute_pars' : { 'epochs': 2,
                   } }

    ######## Run ###########################################
    test_helper(model_pars, data_pars, compute_pars)


def test_helper(model_pars, data_pars, compute_pars):
    global model, session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

    log('Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))
    #
    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.summary())


#######################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire(test)






