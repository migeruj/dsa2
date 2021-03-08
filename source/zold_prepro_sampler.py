# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Transformation for ALL Columns :
   Increase samples, Reduce Samples.

Main isssue is the number of rows change  !!!!

  cannot merge with others

  --> store as train data


  train data ---> new train data


  Transformation with less rows !



2 usage :
    Afte preprocessing, over sample, under-sample.
    


"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, json, copy, numpy as np

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


#### Debuging state (Ture/False)
DEBUG_=True

####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)

def logs(*s):
    if DEBUG_:
        print(*s, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri, load, save_features, params_check
####################################################################################################
####################################################################################################
def pd_export(df, col, pars):
    """
       Export in train folder for next training
       colsall
    :param df:
    :param col:
    :param pars:
    :return:
    """
    colid, colsX, coly = pars['colid'], pars['colsX'], pars['coly']
    dfX   = df[colsX]
    dfX   = dfX.set_index(colid)
    dfX.to_parquet( pars['path_export'] + "/features.parquet")


    dfy = df[coly]
    dfy = dfy.set_index(colid)
    dfX.to_parquet( pars['path_export'] + "/target.parquet")





###################################################################################################
##### Filtering / cleaning rows :   ###############################################################
def pd_filter_rows(df, col, pars):
    """
       Remove rows based on criteria
    :param df:
    :param col:
    :param pars:
    :return:
    """
    import re
    coly = col
    filter_pars =  pars
    def isfloat(x):
        #x = re.sub("[!@,#$+%*:()'-]", "", str(x))
        try :
            a= float(x)
            return 1
        except:
            return 0

    ymin, ymax = pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0)

    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x),axis=1 )
    df = df[ df['_isfloat'] > 0 ]
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]
    del df['_isfloat']
    return df, col


def pd_sample_imblearn(df=None, col=None, pars=None):
    """
        Over-sample
    """
    params_check(pars, ['model_name', 'pars_resample', 'coly']) # , 'dfy'
    prefix = '_sample_imblearn'

    ######################################################################################
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.under_sampling import NearMiss

    # model_resample = { 'SMOTE' : SMOTE, 'SMOTEENN': SMOTEENN }[  pars.get("model_name", 'SMOTEENN') ]
    model_resample = locals()[  pars.get("model_name", 'SMOTEENN')  ]
    pars_resample  = pars.get('pars_resample',
                             {'sampling_strategy' : 'auto', 'random_state':0}) # , 'n_jobs': 2

    if 'path_pipeline' in pars :   #### Inference time
        return df, {'col_new': col }

    else :     ### Training time
        colX    = col # [col_ for col_ in col if col_ not in coly]
        coly    = pars['coly']
        train_y = pars['dfy']  ## df[coly] #
        train_X = df[colX].fillna(method='ffill')
        gp      = model_resample( **pars_resample)
        X_resample, y_resample = gp.fit_resample(train_X, train_y)

        col_new   = [ t + f"_{prefix}" for t in col ] 
        df2       = pd.DataFrame(X_resample, columns = col_new) # , index=train_X.index
        df2[coly] = y_resample

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df2, prefix.replace("col_", "df_"), pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col,            pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_resample,  pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ###  for training input data
    }
    return df2, col_pars




def pd_autoencoder(df, col, pars):
    """"
    (4) Autoencoder
    An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.
    The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction,
    by training the network to ignore noise.
    (i) Feed Forward
    The simplest form of an autoencoder is a feedforward, non-recurrent
    neural network similar to single layer perceptrons that participate in multilayer perceptrons
    """
    from sklearn.preprocessing import minmax_scale
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    def encoder_dataset(df, drop=None, dimesions=20):
        # encode categorical columns
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        print(cat_columns)

        # encode objects columns
        from sklearn.preprocessing import OrdinalEncoder

        def encode_objects(X_train):
            oe = OrdinalEncoder()
            oe.fit(X_train)
            X_train_enc = oe.transform(X_train)
            return X_train_enc

        selected_cols = df.select_dtypes(['object']).columns
        df[selected_cols] = encode_objects(df[selected_cols])

        # df = df[[c for c in df.columns if c not in df.select_dtypes(['object']).columns]]    
        if drop:
            train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
        else:
           train_scaled = minmax_scale(df.values, axis = 0)
        return train_scaled
    # define the number of encoding dimensions
    encoding_dim = pars.get('dimesions', 2)
    # define the number of features
    train_scaled = encoder_dataset(df, pars.get('drop',None), encoding_dim)
    print("train scaled: ", train_scaled)
    ncol = train_scaled.shape[1]
    input_dim = tf.keras.Input(shape = (ncol, ))
    # Encoder Layers
    encoded1      = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
    encoded2      = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
    encoded3      = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
    encoded4      = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
    encoded5      = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
    encoded6      = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
    encoded7      = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)
    encoder       = tf.keras.Model(inputs = input_dim, outputs = encoded7)
    encoded_input = tf.keras.Input(shape = (encoding_dim, ))
    encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
    encoded_train = encoded_train.add_prefix('encoded_')
    if 'drop' in pars :
        drop = pars['drop']
        encoded_train = pd.concat((df[drop],encoded_train),axis=1)

    return encoded_train
    # df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()



def pd_augmentation_sdv(df, col=None, pars={})  :
    '''
    Using SDV Variation Autoencoders, the function augments more data into the dataset
    params:
            df          : (pandas dataframe) original dataframe
            col : column name for data enancement
            pars        : (dict - optional) contains:                                          
                n_samples     : (int - optional) number of samples you would like to add, defaul is 10%
                primary_key   : (String - optional) the primary key of dataframe
                aggregate  : (boolean - optional) if False, prints SVD metrics, else it averages them
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved
    returns:
            df_new      : (pandas dataframe) df with more augmented data
            col         : (list of strings) same columns
    '''
    n_samples       = pars.get('n_samples', max(1, int(len(df) * 0.10) ) )   ## Add 10% or 1 sample by default value
    primary_key     = pars.get('colid', None)  ### Custom can be created on the fly
    metrics_type    = pars.get('aggregate', False)
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    model_name      = pars.get('model_name', "TVAE")
    
    # importing libraries
    try:
        #from sdv.demo import load_tabular_demo
        from sdv.tabular import TVAE
        from sdv.tabular import CTGAN
        from sdv.timeseries import PAR
        from sdv.evaluation import evaluate
        import ctgan
        
        if ctgan.__version__ != '0.3.1.dev0':
            raise Exception('ctgan outdated, updating...')
    except:
        os.system("pip install sdv")
        os.system('pip install ctgan==0.3.1.dev0')
        from sdv.tabular import TVAE
        from sdv.tabular import CTGAN
        from sdv.timeseries import PAR
        from sdv.evaluation import evaluate      
    

    # model fitting 
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')
                
            model = {'TVAE' : TVAE, 'CTGAN' : CTGAN, 'PAR' : PAR}[model_name]
            if model_name == 'PAR':
                model = model(entity_columns = pars['entity_columns'],
                              context_columns = pars['context_columns'],
                              sequence_index = pars['sequence_index'])
            else:
                model = model(primary_key=primary_key)   
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ', path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)

    log('##### Generating Samples #############')
    new_data = model.sample(n_samples)
    log_pd( new_data, n=7)
    
   
    log('######### Evaluation Results #########')
    if metrics_type == True:
      evals = evaluate(new_data, df, aggregate= True )        
      log(evals)
    else:
      evals = evaluate(new_data, df, aggregate= False )        
      log_pd(evals, n=7)
    
    # appending new data    
    df_new = df.append(new_data)
    log(str(len(df_new) - len(df)) + ' new data added')
    
    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' ) 
        log('###### df augmentation save on disk', pars['path_newdata'] )    
    
    log('###### augmentation complete ######')
    return df_new, col


#####################################################################################
#####################################################################################
def test_pd_augmentation_sdv():
    from sklearn.datasets import load_boston
    data = load_boston()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    log_pd(df)

    dir_tmp = 'ztmp/'     
    path = dir_tmp + '/model_par_augmentation.pkl'
    os.makedirs(dir_tmp, exist_ok=True)

    log('##### testing augmentation CTGAN ######################')
    pars = {'path_model_save': path,  'model_name': 'CTGAN'}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)

    log('####### Reload')
    df_new, _ = pd_augmentation_sdv(df, pars={'path_model_load': path})
    

    log('##### testing augmentation VAE #########################')
    pars = {'path_model_save': path, 'model_name': 'VAE'}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    log('####### Reload')
    df_new, _ = pd_augmentation_sdv(df, pars={'path_model_load': path})


    log('##### testing Time Series #############################')
    from sdv.demo import load_timeseries_demo        
    df = load_timeseries_demo()
    log_pd(df)

    entity_columns  = ['Symbol']
    context_columns = ['MarketCap', 'Sector', 'Industry']
    sequence_index  = 'Date'

    pars = {'path_model_save': path,
            'model_name': 'PAR',
            'entity_columns' : entity_columns,
            'context_columns': context_columns,
            'sequence_index' : sequence_index,
            'n_samples' : 5}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    
    log('####### Reload')
    df_new, _ = pd_augmentation_sdv(df, pars={'path_model_load': path,  'n_samples' : 5 })
    log_pd(df_new)



def pd_covariate_shift_adjustment():
    """
    https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766
     Covariate shift has been extensively studied in the literature, and a number of proposals to work under it have been published. Some of the most important ones include:
        Weighting the log-likelihood function (Shimodaira, 2000)
        Importance weighted cross-validation (Sugiyama et al, 2007 JMLR)
        Integrated optimization problem. Discriminative learning. (Bickel et al, 2009 JMRL)
        Kernel mean matching (Gretton et al., 2009)
        Adversarial search (Globerson et al, 2009)
        Frank-Wolfe algorithm (Wen et al., 2015)
    """
    import numpy as np
    from scipy import sparse
    import pylab as plt

    # .. to generate a synthetic dataset ..
    from sklearn import datasets
    n_samples, n_features = 1000, 10000
    A, b = datasets.make_regression(n_samples, n_features)
    def FW(alpha, max_iter=200, tol=1e-8):
        # .. initial estimate, could be any feasible point ..
        x_t = sparse.dok_matrix((n_features, 1))
        trace = []  # to keep track of the gap
        # .. some quantities can be precomputed ..
        Atb = A.T.dot(b)
        for it in range(max_iter):
            # .. compute gradient. Slightly more involved than usual because ..
            # .. of the use of sparse matrices ..
            Ax = x_t.T.dot(A.T).ravel()
            grad = (A.T.dot(Ax) - Atb)
            # .. the LMO results in a vector that is zero everywhere except for ..
            # .. a single index. Of this vector we only store its index and magnitude ..
            idx_oracle = np.argmax(np.abs(grad))
            mag_oracle = alpha * np.sign(-grad[idx_oracle])
            g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
            trace.append(g_t)
            if g_t <= tol:
                break
            q_t = A[:, idx_oracle] * mag_oracle - Ax
            step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
            x_t = (1. - step_size) * x_t
            x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
        return x_t, np.array(trace)

    # .. plot evolution of FW gap ..
    sol, trace = FW(.5 * n_features)
    plt.plot(trace)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('FW gap')
    plt.title('FW on a Lasso problem')
    plt.grid()
    plt.show()
    sparsity = np.mean(sol.toarray().ravel() != 0)
    print('Sparsity of solution: %s%%' % (sparsity * 100))




########################################################################################
########################################################################################
def test():
    from util_feature import test_get_classification_data
    dfX, dfy = test_get_classification_data()
    cols     = list(dfX.columsn)
    ll       = [ ('pd_sample_imblearn', {}  )


               ]

    for fname, pars in ll :
        myfun = globals()[fname]
        res   = myfun(dfX, cols, pars)



if __name__ == "__main__":
    import fire
    fire.Fire()
    
    
    
    
    
    
    
    
"""


def pd_generic_transform(df, col=None, pars={}, model=None)  :
 
     Transform or Samples using  model.fit()   model.sample()  or model.transform()
    params:
            df    : (pandas dataframe) original dataframe
            col   : column name for data enancement
            pars  : (dict - optional) contains:                                          
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved 
    returns:
            model, df_new, col, pars
   
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    pars_model      = pars.get('pars_model', {} )
    model_method    = pars.get('method', 'transform')
    
    # model fitting 
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')
            model = model( **pars_model)
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ' + path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)

    log('##### Generating Samples/transform #############')    
    if model_method == 'sample' :
        n_samples =pars.get('n_samples', max(1, 0.10 * len(df) ) )
        new_data  = model.sample(n_samples)
        
    elif model_method == 'transform' :
        new_data = model.transform(df.values)
    else :
        raise Exception("Unknown", model_method)
        
    log_pd( new_data, n=7)    
    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' ) 
        log('###### df transform save on disk', pars['path_newdata'] )    
    
    return model, df_new, col, pars



"""
    
    
    
