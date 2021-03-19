# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
# coding: utf-8

# # VAEM: a Deep Generative Model for Heterogeneous Mixed Type Data

# VAEM is an extension of variational autoencoders (VAEs) in order to handle such heterogeneous data. It is a deep generative model that is trained in a two stage manner. In the first stage we fit a different VAE independently to each data dimension $x_{nd}$. We call the resulting $D$ models marginal VAEs. Then, in the second stage, in order to capture the inter-variable dependencies, a new multi-dimensional VAE, called the dependency network, is build on top of the latent representations provided by the first-stage encoders. Finally, if the model is used in down stream tasks such as sequential active information acquisition, we often introduce a third stage, which is to add a new discriminator (preditor) model on top of the VAEM outputs. 
# 
# Different stages are referred as `1`,`2`, and `3` in `list_stages` in the `.json` files. 

# ## Usage

# To run the demo, you need to first download the [Bank Marketing UCI dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), 
# and put the csv file under `data/bank`. You will need to preprocess the data into the format according to our example .csv 
# file (which does not contain any real data). This can done by splitting the text into columns using `;` as delimiters. Then,
# simply run `Main_Notebook.ipynb`. This notebook train/or load a VAEM model on Bank dataset, 
# and demonstrates how to perform sequential active information acquisition (SAIA) and imputation. 
# By default, it trains a new model on Bank dataset. If you would like to load a pre-trained model, by default it will load a pre-trained tensorflow model from `saved_weights/bank/`. Note that in order to perform 
# active information acquisition, an additional third stage training is required. This will add a discriminator (predictor)
# to the model, which is required for SAIA. The configurations for VAEM can be found in `.json` files in `hyperparameters/bank`, 
# which include:
# * "list_stage" : list of stages that you would like the model to be trained. stage 1 = training marginal VAEs, stage 2 = training dependency network,  stage 3 = add predictor and improve predictive performance. The default is [1,2]. 
# * "epochs" : number of epochs for training VAEM. If you would like to load a pretrained model rather than training a new one, you can simply set this to zero.
# * "latent_dim" : size of latent dimensions of dependency network, 
# * "p" : upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability. Our suggestion is that if original dataset already contains missing data, you can just set p to 0. 
# * "iteration" : iterations (number of mini batches) used per epoch. set to -1 to run the full epoch. If your dataset is large, please set to other values such as 10.
# * "batch_size" : iterations (number of mini batches) used per epoch. set to -1 to run the full epoch. If your dataset is large, please set to other values such as 10.
# * "K" : the dimension of the feature map (h) dimension of PNP encoder.
# * "M" : Number of MC samples when perform imputing. 
# * "repeat" : number of repeats.
# * "data_name" : name of the dataset being used. Our default is "bank".
# * "output_dir" : Directory where the model is stored. Our default is "./saved_weights/bank/",
# * "data_dir" : Directory where the data is stored. Our default is "./data/bank/",
# * "list_strategy" : list of strategies for active learning, 0 = random, 1 = single ordering. Default: [1]

# ## Load modules
"""
import numpy as np, sys,os
import tensorflow as tf
print(tf.__version__)
from scipy.stats import bernoulli
import os, copy
import random
from random import sample
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
plt.switch_backend('agg')
# tfd = tf.contrib.distributions
import json
import seaborn as sns; sns.set(style="ticks", color_codes=True)




#############################################################################################################################
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/repo/VAEM/" )

import repo.VAEM.utils.process as process
import repo.VAEM.utils.params as params
import repo.VAEM.utils.active_learning as active_learning






#######################################################################################
verbosity =2

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 :
      print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None



####################################################################################################
##### Custom code  #################################################################################
cols_ref_formodel = ['none']  ### No column group


max_Data = 0.7
min_Data = 0.3

def encode2(data_decode,list_discrete,records_d,fast_plot):
    args = params.Params('repo/VAEM/hyperparameters/bank_plot.json')
    Data_train_decomp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode
    vae = active_learning.p_vae_active_learning(Data_train_decomp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type,args,list_discrete,records_d)
    tf.reset_default_graph()
    ### Impute missing data. Fthe mask to be zeros

    x_recon,z_posterior,x_recon_cat_p = model.get_imputation( Data_train_noisy_decomp, mask_train_decomp*0,cat_dims,dic_var_type) ## one hot already cpverted to integer


    x_real = process.compress_data(Data_train_decomp,cat_dims, dic_var_type) ## x_real still needs conversion
    x_real_cat_p = Data_train_decomp[:,0:(cat_dims.sum()).astype(int)]


    # max_Data = 0.7
    # min_Data = 0.3
    Data_std = (x_real - x_real.min(axis=0)) / (x_real.max(axis=0) - x_real.min(axis=0))
    scaling_factor = (x_real.max(axis=0) - x_real.min(axis=0))/(max_Data - min_Data)
    Data_real = Data_std * (max_Data - min_Data) + min_Data
    sub_id = [1,2,10]

    if fast_plot ==1:
        Data_real = pd.DataFrame(Data_real[:,sub_id])
        g = sns.pairplot(Data_real.sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    else:
        Data_real = pd.DataFrame(Data_real[:,sub_id])
        g = sns.pairplot(Data_real.sample(min(10000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g = g.map_upper(plt.scatter,marker='+')
        g = g.map_lower(sns.kdeplot, cmap="hot",shade=True,bw=.1)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    Data_fake_noisy= x_recon
    Data_fake = process.invert_noise(Data_fake_noisy,list_discrete_comp,records_d)

    Data_std = (Data_fake - x_real.min(axis=0)) / (x_real.max(axis=0) - x_real.min(axis=0))
    Data_fake = Data_std * (max_Data - min_Data) + min_Data


    sub_id = [1,2,10]

    if fast_plot ==1:
        g = sns.pairplot(pd.DataFrame(Data_fake[:,sub_id]).sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    else:
        g = sns.pairplot(pd.DataFrame(Data_fake[:,sub_id]).sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g = g.map_upper(plt.scatter,marker='+')
        g = g.map_lower(sns.kdeplot, cmap="hot",shade=True,bw=.1)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))

    return model, scaling_factor


def decode2(data_decode,scaling_factor,list_discrete,records_d):
    args = params.Params('repo/VAEM/hyperparameters/bank_SAIA.json')
    Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode
    vae = active_learning.p_vae_active_learning(Data_train_comp, Data_train_noisy_decomp,
        mask_train_decomp, Data_test_decomp,
        mask_test_comp,mask_test_decomp,
        cat_dims,
        DIM_FLT,dic_var_type,
        args,list_discrete,records_d)



    npzfile = np.load(args.output_dir+'/UCI_rmse_curve_SING.npz')
    IC_SING=npzfile['information_curve']*scaling_factor[-1]
    plt.figure(0)
    L = IC_SING.shape[1]
    fig, ax1 = plt.subplots()
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = [0.45, 0.4, 0.45, 0.45]
    ax1.plot(np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).mean(axis=0),'ys',linestyle = '-.', label = 'VAEM-Bank dataset')
    ax1.errorbar(np.arange(IC_SING.shape[2]),np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).mean(axis=0), yerr=np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).std(axis = 0)/np.sqrt(IC_SING.shape[0]),ecolor='y',fmt = 'ys')
    plt.xlabel('Steps',fontsize=18)
    plt.ylabel('avg. test. RMSE',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(bbox_to_anchor=(0.0, 1.02, 1., .102), mode = "expand", loc=3,
               ncol=1, borderaxespad=0.,prop={'size': 20}, frameon=False)
    ax1.ticklabel_format(useOffset=False)
    plt.show()

    return vae


def save_model2(model,output_dir):
    model.save(output_dir)



def test2(fast_plot=0):
    """
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    rs = 42 # random seed
    # the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
    max_Data = 0.7
    min_Data = 0.3
    list_discrete = np.array([8,9])

    # list of categorical variables
    list_cat = np.array([0,1,2,3,4,5,6,7])
    # list of numerical variables
    list_flt = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
    data_decode,records_d = load_data(list_flt,list_cat,list_discrete,max_Data,min_Data,rs)
    """
    list_discrete = np.array([8,9])
    args = params.Params('repo/VAEM/hyperparameters/bank_plot.json')
    data_decode,records_d = load_data()
    Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode

    model,scaling_factor = encode2(data_decode,list_discrete,records_d,fast_plot)
    model = decode2(data_decode, scaling_factor,list_discrete,records_d)

    save_model2(model, args.output_dir)


def load_data(): #(list_flt,list_cat,list_discrete,max_Data,min_Data,rs):
    args = params.Params('repo/VAEM/hyperparameters/bank_plot.json')

    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    rs = 42 # random seed
    # the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
    max_Data = 0.7
    min_Data = 0.3
    list_discrete = np.array([8,9])

    # list of categorical variables
    list_cat = np.array([0,1,2,3,4,5,6,7])
    # list of numerical variables
    list_flt = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
    #data_decode,records_d = load_data(list_flt,list_cat,list_discrete,max_Data,min_Data,rs)

    seed = 100
    bank_raw = pd.read_csv("repo/VAEM/data/bank/bankmarketing_train.csv")
    print(bank_raw.info())
    label_column="y"
    matrix1 = bank_raw.copy()
    print(matrix1.shape)
    process.encode_catrtogrial_column(matrix1, ["job"])
    process.encode_catrtogrial_column(matrix1, ["marital"])
    process.encode_catrtogrial_column(matrix1, ["education"])
    process.encode_catrtogrial_column(matrix1, ["default"])
    process.encode_catrtogrial_column(matrix1, ["housing"])
    process.encode_catrtogrial_column(matrix1, ["loan"])
    process.encode_catrtogrial_column(matrix1, ["contact"])
    process.encode_catrtogrial_column(matrix1, ["month"])
    process.encode_catrtogrial_column(matrix1, ["day_of_week"])
    process.encode_catrtogrial_column(matrix1, ["poutcome"])
    process.encode_catrtogrial_column(matrix1, ["y"])

    Data = ((matrix1.values).astype(float))[0:,:]
    list_discrete_in_flt = (np.in1d(list_flt, list_discrete).nonzero()[0])
    list_discrete_comp = list_discrete_in_flt + len(list_cat)

    if len(list_flt)>0 and len(list_cat)>0:
        list_var = np.concatenate((list_cat,list_flt))
    elif len(list_flt)>0:
        list_var = list_flt
    else:
        list_var = list_cat
    Data_sub = Data[:,list_var]
    dic_var_type = np.zeros(Data_sub.shape[1])
    dic_var_type[0:len(list_cat)] = 1

    # In this notebook we assume the raw data matrix is fully observed
    Mask = np.ones(Data_sub.shape)
    # Normalize/squash the data matrix
    Data_std = (Data_sub - Data_sub.min(axis=0)) / (Data_sub.max(axis=0) - Data_sub.min(axis=0))
    scaling_factor = (Data_sub.max(axis=0) - Data_sub.min(axis=0))/(max_Data - min_Data)
    Data_sub = Data_std * (max_Data - min_Data) + min_Data

    # decompress categorical data into one hot representation
    Data_cat = Data[:,list_cat].copy()
    Data_flt = Data[:,list_flt].copy()
    Data_comp = np.concatenate((Data_cat,Data_flt),axis = 1)
    Data_decomp, Mask_decomp, cat_dims, DIM_FLT = process.data_preprocess(Data_sub,Mask,dic_var_type)
    Data_train_decomp, Data_test_decomp, mask_train_decomp, mask_test_decomp,mask_train_comp, mask_test_comp,Data_train_comp, Data_test_comp = train_test_split(
            Data_decomp, Mask_decomp,Mask,Data_comp,test_size=0.1, random_state=rs)

    list_discrete = list_discrete_in_flt + (cat_dims.sum()).astype(int)

    Data_decomp = np.concatenate((Data_train_decomp, Data_test_decomp), axis=0)
    Data_train_orig = Data_train_decomp.copy()
    Data_test_orig = Data_test_decomp.copy()

    # Note that here we have added some noise to continuous-discrete variables to help training. Alternatively, you can also disable this by changing the noise ratio to 0.
    Data_noisy_decomp,records_d, intervals_d = process.noisy_transform(Data_decomp, list_discrete, noise_ratio = 0.99)
    noise_record = Data_noisy_decomp - Data_decomp
    Data_train_noisy_decomp = Data_noisy_decomp[0:Data_train_decomp.shape[0],:]
    Data_test_noisy_decomp = Data_noisy_decomp[Data_train_decomp.shape[0]:,:]
    
    
    data_decode = (Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type)
    
    return data_decode,records_d



##################################################################################################
##################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
            return
        # log2("data_pars", data_pars)
        model_class = model_pars['model_class']  #

        ### get model params  #######################################################
        mdict_default = {
             'class_num':           5
            ,'intermediate_dim':    64
        }
        mdict = model_pars.get('model_pars', mdict_default)

        ### Dynamic Dimension : data_pars  ---> model_pars dimension  ###############
        data_decode,records_d = load_data()




        #### Model setup ################################
        list_discrete = np.array([8,9])
        args = params.Params('./hyperparameters/bank_plot.json')
        model1,scaling_factor = encode2(data_decode,list_discrete,records_d,fast_plot)
        model2                = decode2(data_decode, scaling_factor,list_discrete,records_d)

        self.model ={
            'encode' :  model1,
            'decode' :  model2
        }

        log2(self.model_pars, self.model)
        self.model.summary()


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train",)
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle

    ### Fake label
    Xtrain_dummy = np.ones((Xtrain_tuple.shape[0], 1))

    assert 'epochs' in cpars, 'epoch missing'
    model.model['encode'].fit([Xtrain_tuple, Xtrain_dummy],  **cpars)
    model.model['decode'].fit([Xtrain_tuple, Xtrain_dummy],  **cpars)



def encode(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        # data_pars['train'] = False
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    Xdummy = np.ones((Xpred_tuple.shape[0], 1))
    ypred = model.model.encode([Xpred_tuple, Xdummy ] )

    return ypred


def decode(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        # data_pars['train'] = False
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    Xdummy = np.ones((Xpred_tuple.shape[0], 1))
    ypred = model.model.decode([Xpred_tuple, Xdummy ] )

    return ypred



####################################################################################################
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :
        return Xtrain

    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )

    return Xtuple_train



def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    if data_pars.get('dataset_name', '') == 'correlation' :
       x_train, ytrain = load_data(data_pars)
       return x_train, ytrain


    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

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
            #flog2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')

######################################################################################
def reset():
    global model, session
    model, session = None, None

def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

    model.model.save(f"{path}/model_keras.h5")
    model.model.save_weights(f"{path}/model_keras_weights.h5")

    modelx = Model()  # Empty model  Issue with pickle
    modelx.model_pars   = model.model_pars
    modelx.data_pars    = model.data_pars
    modelx.compute_pars = model.compute_pars
    # log('model', modelx.model)
    pickle.dump(modelx, open(f"{path}/model.pkl", mode='wb'))  #

    pickle.dump(info, open(f"{path}/info.pkl", mode='wb'))  #


def load_model(path=""):
    global model, session
    import dill as pickle

    model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model        = get_model( model0.model_pars)
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars

    model.model.load_weights( f'{path}/model_keras_weights.h5')

    log(model.model.summary())
    #### Issue when loading model due to custom weights, losses, Keras erro
    #model_keras = get_model()
    #model_keras = keras.models.load_model(path + '/model_keras.h5' )
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



##################################################################################

def test():
    ### Custom dataset
    adata_pars = {'dataset_name':  'correlation'}
    adata_pars['state_num']           = 10
    X,y = load_data()


    ####
    d = {'task_type' : 'train', 'data_type': 'ram',}
    d['signal_dimension'] = 15

    d["train"] ={
      "Xtrain":  X[:10,:],
      "ytrain":  y[:10,:],        ## Not used
      "Xtest":   X[10:1000,:],
      "ytest":   y[10:1000,:],    ## Nor Used
    }

    data_pars= d
    m                       = {}
    m['original_dim']       = np.uint32( adata_pars['signal_dimension']*(adata_pars['signal_dimension']-1)/2)
    m['class_num']          = 5
    model_pars = {'model_pars'  : m,
                  'model_class' : "class_VAEMDN"
                 }

    compute_pars = {}
    compute_pars['compute_pars'] = {'epochs': 1, }   ## direct feed


    ### Meta Class #########################################################
    X = load_data()
    test_helper(model_pars, data_pars, compute_pars, X)



def test_helper(model_pars, data_pars, compute_pars, Xpred):
    global model, session
    init()
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)


    log('Predict data..')
    ypred, ypred_proba = encode(Xpred=Xpred, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:3]}')

    #
    log('Saving model..')
    log( model.model )
    save(path= root + '/model_dir/')


    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.model)

    log('Predict data..')
    ypred, ypred_proba = encode(Xpred=Xpred, data_pars=data_pars, compute_pars=compute_pars)







if __name__ == "__main__":
    test2()



