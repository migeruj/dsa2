# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
ipython source/models/keras_widedeep.py  test  --pdb


python keras_widedeep.py  test

pip install Keras==2.4.3


"""
import os, pandas as pd, numpy as np, sklearn, copy
from sklearn.model_selection import train_test_split

from keras.layers import Lambda, Input, Dense, Reshape
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import numpy as np
import tensorflow

try :
  import keras
  from keras.callbacks import EarlyStopping, ModelCheckpoint
  from keras import layers
except :
  from tensorflow import keras
  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
  from tensorflow.keras import layers


####################################################################################################
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
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


cols_ref_formodel = ['none']  ### No column group
def get_model(model_pars):
    
    original_dim       = model_pars['original_dim']
    class_num          = model_pars['class_num']
    intermediate_dim   = model_pars['intermediate_dim']
    intermediate_dim_2 = model_pars['intermediate_dim_2']
    latent_dim         = model_pars['latent_dim']
    Lambda1            = model_pars['Lambda1']
    batch_size         = model_pars['batch_size']
    Lambda2            = model_pars['Lambda2']
    Alpha              = model_pars['Alpha']
    
    input_shape = (original_dim, )
    inputs = Input(shape=input_shape, name='encoder_input')
    inter_x1 = Dense(intermediate_dim, activation='tanh',
                     name='encoder_intermediate')(inputs)
    inter_x2 = Dense(intermediate_dim_2, activation='tanh',
                     name='encoder_intermediate_2')(inter_x1)
    inter_x3 = Dense(intermediate_dim_2, activation='tanh',
                     name='encoder_intermediate_3')(inter_x1)
    # add 3 means as additional parameters
    dummy = Input(shape=(1,), name='dummy')
    mu_vector = Dense(class_num*latent_dim, name='mu_vector',
                      use_bias=False)(dummy)
    mu = Reshape((class_num, latent_dim), name='mu')(mu_vector)

    # prior categorical distribution
    pi = Dense(class_num, activation='softmax', name='pi')(dummy)

    # posterior categorical distribution
    c = Dense(class_num, activation='softmax', name='c')(inter_x2)

    # outlier/non-outlier classification (Posterior Beta)
    # inter_outlier = Dense(128, activation='relu', name='inter_outlier')(x)
    c_outlier = Dense(2, activation='softmax', name='c_outlier')(inter_x3)

    # q(z|x)
    z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
    z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = keras.models.Model([inputs, dummy], [z_mean, z_log_var, z,
                                      mu, c, c_outlier, pi], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    inter_y1 = Dense(intermediate_dim_2, activation='tanh')(latent_inputs)
    inter_y2 = Dense(intermediate_dim, activation='tanh')(inter_y1)
    outputs = Dense(original_dim, activation='tanh')(inter_y2)

    # instantiate decoder model
    decoder = keras.models.Model(latent_inputs, outputs, name='decoder')
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder([inputs, dummy])[2])
    vae = keras.models.Model([inputs, dummy], outputs, name='vae_mlp')
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = tf.compat.v1.multiply(
        reconstruction_loss, c_outlier[:, 0])
    reconstruction_loss *= original_dim
    kl_loss_all = tf.compat.v1.get_variable("kl_loss_all", [batch_size, 1],
                                    dtype=tf.compat.v1.float32, initializer=tf.compat.v1.zeros_initializer)
    kl_cat_all = tf.compat.v1.get_variable("kl_cat_all", [batch_size, 1],
                                   dtype=tf.compat.v1.float32, initializer=tf.compat.v1.zeros_initializer)
    dir_prior_all = tf.compat.v1.get_variable("dir_prior_all", [batch_size, 1],
                                      dtype=tf.compat.v1.float32, initializer=tf.compat.v1.zeros_initializer)
    for i in range(0, class_num):
        c_inlier = tf.compat.v1.multiply(c[:, i], c_outlier[:, 0])

        # kl-divergence between q(z|x) and p(z|c)
        kl_loss = 1 + z_log_var - \
            K.square(z_mean-mu[:, i, :]) - K.exp(z_log_var)
        kl_loss = tf.compat.v1.multiply(K.sum(kl_loss, axis=-1), c_inlier)
        kl_loss *= -0.5
        kl_loss_all = kl_loss_all + kl_loss

        # kl-divergence between q(c|x) and p(c) (not including outlier class)
        mc = K.mean(c[:, i])
        mpi = K.mean(pi[:, i])
        kl_cat = mc * K.log(mc) - mc * K.log(mpi)
        kl_cat_all = kl_cat_all + kl_cat

        # Dir prior: Dir(3, 3, ..., 3)
        dir_prior = -0.1*K.log(pi[:, i])
        dir_prior_all = dir_prior_all+dir_prior
    mco1 = K.mean(c_outlier[:, 0])
    mco2 = K.mean(c_outlier[:, 1])
    mpo1 = 1-Alpha
    mpo2 = Alpha
    kl_cat_outlier = (mco1 * K.log(mco1) - mco1 * np.log(mpo1) +
                      mco2 * K.log(mco2) - mco2 * np.log(mpo2))

    # total loss
    vae_loss = K.mean(reconstruction_loss +
                      kl_loss_all +
                      dir_prior_all +
                      Lambda1*kl_cat_all)+Lambda2*kl_cat_outlier

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae


##################################################################################################
##################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
            return

        log2("data_pars", data_pars)
        model_class = model_pars['model_class']  #

        ### Dynamic shape of input
        #model_pars['model_pars']['n_feat']       = model_pars['model_pars']['n_deep']
        mdict_default = {
             'original_dim':        np.uint32( data_pars['signal_dimension']*(data_pars['signal_dimension']-1)/2)
            ,'class_num':           5
            ,'intermediate_dim':    64
            ,'intermediate_dim_2':  16
            ,'latent_dim':          3
            ,'Lambda1':             1
            ,'batch_size':          256
            ,'Lambda2':             200
            ,'Alpha':               0.075
        }
        mdict = model_pars.get('model_pars', mdict_default)

        self.model  = get_model(mdict)
        log2(model_class, self.model)
        self.model.summary()


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train",)
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle

    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model_ckpt     = ModelCheckpoint(filepath = compute_pars.get('path_checkpoint', 'ztmp_checkpoint/'),
                                     save_best_only=True, monitor='loss')
    cpars['callbacks'] =  [early_stopping, model_ckpt]

    ### Fake label
    #ytrain = ytrain.reshape(ytrain.shape[0],-1)
    #ytest = ytest.reshape(ytest.shape[0],-1)
    ytrain = np.ones((Xtrain_tuple.shape[0], 1))
    assert 'epochs' in cpars, 'epoch missing'
    hist = model.model.fit([Xtrain_tuple, ytrain],
                           # validation_data=(Xtest_tuple, ytest),
                            **cpars)
    model.history = hist


def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        # data_pars['train'] = False
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    ypred = model.model.predict(Xpred_tuple )

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba



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
       x_train, ytrain = get_mydata_correl(data_pars)
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
            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



def get_label(encoder, x_train, dummy_train, class_num=5, batch_size=256):
    [z_mean, z_log_var, z, mu, c, c_outlier, pi] = encoder.predict(
        [x_train, dummy_train], batch_size=batch_size)

    labels = np.zeros(x_train.shape[0])
    for i in range(0, x_train.shape[0]):
        max_prob = np.max(np.multiply(c[i, :], c_outlier[i, 0]))
        idx = np.argmax(np.multiply(c[i, :], c_outlier[i, 0]))
        if (max_prob > c_outlier[i, 1]):
            labels[i] = idx
        else:
            labels[i] = class_num
    return labels


######################################################################################
def reset():
    global model, session
    model, session = None, None

def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

    model.model.save(f"{path}/model_keras.h5")

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

    model_keras = keras.models.load_model(path + '/model_keras.h5' )
    model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model_keras
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



##################################################################################
def get_mydata_correl(data_pars):
    state_num = data_pars['state_num']
    time_len = data_pars['time_len']
    signal_dimension = data_pars['signal_dimension']
    CNR = data_pars['CNR']
    window_len = data_pars['window_len']
    half_window_len = data_pars['half_window_len']
    a = np.ones(shape=(state_num, state_num))
    alpha = np.ones(10)*10
    alpha[5:] = 1
    base_prob = np.random.dirichlet(alpha) * 0.1
    for t in range(state_num):
        a[t, :] = base_prob
        a[t, t] += 0.9

    # simulate states
    state = np.zeros(time_len, dtype=np.uint8)
    p = np.random.uniform()
    state[0] = np.floor(p*state_num)
    for t in range(0, time_len-1):
        p = np.random.uniform()
        for s in range(state_num):
            if (p <= np.sum(a[state[t], :s+1])):
                state[t+1] = s
                break

    freq = np.zeros(state_num)
    for t in range(state_num):
        freq[t] = np.sum(state == t)
    loading = np.random.randint(-1, 2, size=(state_num, signal_dimension))

    cov = np.zeros((state_num, signal_dimension, signal_dimension))
    for t in range(state_num):
        cov[t, :, :] = np.matmul(np.transpose(
            [loading[t, :]]), [loading[t, :]])

    # generate BOLD signal
    signal = np.zeros((time_len, signal_dimension))
    for t in range(0, time_len):
        signal[t, :] = np.random.multivariate_normal(
            np.zeros((signal_dimension)), cov[state[t], :, :])
    signal += np.random.normal(size=signal.shape)/CNR
    original_dim = np.uint32(signal_dimension*(signal_dimension-1)/2)

    x_train = np.zeros(
        shape=(time_len-window_len*2, np.uint32(original_dim)))
    sum_corr = np.zeros(shape=(state_num, original_dim))
    occupancy = np.zeros(state_num)

    for t in range(window_len, time_len-window_len):
        corr_matrix = np.corrcoef(np.transpose(
            signal[t-half_window_len:t+half_window_len+1, :]))
        upper = corr_matrix[np.triu_indices(signal_dimension, k=1)]
        x_train[t-window_len, :] = np.squeeze(upper)
        if (np.sum(state[t-half_window_len:t+half_window_len+1] == state[t]) == window_len):
            sum_corr[state[t], :] += x_train[t-window_len, :]
            occupancy[state[t]] += 1


    y_train = np.ones((x_train.shape[0], 1))
    return x_train, y_train


def test():
    ### Custom dataset
    adata_pars = {'dataset_name':  'correlation'}
    adata_pars['state_num']           = 10
    adata_pars['time_len']            = 50000
    adata_pars['signal_dimension']    = 15
    adata_pars['CNR']                 = 1
    adata_pars['window_len']          = 11
    adata_pars['half_window_len']     = 5
    X,y = get_mydata_correl(adata_pars)


    ####
    d = {'task_type' : 'train', 'data_type': 'ram',}
    d['signal_dimension'] = 15

    d["train"] ={
      "Xtrain":  X[:100,:],
      "ytrain":  y[:100,:],
      "Xtest":   X[100:1000,:],
      "ytest":   y[100:1000,:],
    }

    data_pars= d


    model_pars                       = {}
    model_pars['original_dim']       = np.uint32( adata_pars['signal_dimension']*(adata_pars['signal_dimension']-1)/2)
    model_pars['class_num']          = 5
    model_pars['intermediate_dim']   = 64
    model_pars['intermediate_dim_2'] = 16
    model_pars['latent_dim']         = 3
    model_pars['Lambda1']            = 1
    model_pars['batch_size']         = 256
    model_pars['Lambda2']            = 200
    model_pars['Alpha']              = 0.075
    model_pars['model_pars'] = model_pars
    model_pars['model_class'] = "Kears"

    compute_pars = {}
    compute_pars['compute_pars'] = {'epochs': 1, }   ## direct feed


    ### Meta Class #########################################################
    Xpred,_ = get_mydata_correl(adata_pars)
    test_helper(model_pars, data_pars, compute_pars, Xpred)



def test_helper(model_pars, data_pars, compute_pars, Xpred):
    global model, session
    init()
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=Xpred, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

    #
    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.summary())






if __name__ == "__main__":
    test()




