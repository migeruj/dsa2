# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
colnum, colcat, coldate transformation

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


from util_feature import  (save,  load, save_features, os_get_function_name,
                           params_check)
import util_feature


####################################################################################################
####################################################################################################
def pd_col_atemplate(df: pd.DataFrame, col: list=None, pars: dict=None):
    """
    Example of custom Processor
    Used at prediction time
        "path_pipeline"  : 

    Training time :
        "path_features_store" :  to store intermediate dataframe
        "path_pipeline_export":  to store pipeline  for later usage

    """
    from source.util_feature import save, load
    prefix = "col_myfun"
    #### Inference time LOAD previous pars  ########################################### 
    if "path_pipeline" in pars :   
        prepro   = load(pars["path_pipeline"] + f"/{prefix}_model.pkl" )
        pars     = load(pars["path_pipeline"] + f"/{prefix}_pars.pkl" )
        pars     = {} if pars is None else  pars

    #### Do something #################################################################
    df_new         = df[col]  ### Do nithi
    df_new.columns = [  col + "_myfun"  for col in df.columns ]
    cols_new       = list(df_new.columns)

    prepro   = None   ### model
    pars_new = None   ### new params



    ###################################################################################
    ###### Training time save all #####################################################
    if "path_features_store" in pars and "path_pipeline_export" in pars :
       save(prepro,         pars["path_pipeline_export"] + f"/{prefix}_model.pkl" )
       save(cols_new,       pars["path_pipeline_export"] + f"/{prefix}.pkl" )
       save(pars_new,       pars["path_pipeline_export"] + f"/{prefix}_pars.pkl" )

    ###### Training & Inference time : df + new column names ##########################
    col_pars = {"prefix" : prefix , "path" :   pars.get("path_pipeline_export", pars.get("path_pipeline", None)) }
    col_pars["cols_new"] = {
        "col_myfun" :  cols_new  ### new column list
    }
    return df_new, col_pars



###########################################################################################
##### Label processing   ##################################################################
def pd_coly_clean(df: pd.DataFrame, col: list=None, pars: dict=None):
    path_features_store = pars['path_features_store']
    # path_pipeline_export = pars['path_pipeline_export']
    coly = col=[0]
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )
        save_features(df[coly], 'dfy', path_features_store)
    return df,coly


def pd_coly(df: pd.DataFrame, col: list=None, pars: dict=None):
    ##### Filtering / cleaning rows :   ################
    coly=col
    def isfloat(x):
        try :
            a= float(x)
            return 1
        except:
            return 0
    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x) )
    df             = df[ df['_isfloat'] > 0 ]
    df[coly]       = df[coly].astype('float64')
    del df['_isfloat']
    logs("----------df[coly]------------",df[coly])
    ymin, ymax = pars.get('ymin', -9999999999.0), pars.get('ymax', 999999999.0)
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]

    ##### Label processing   ####################################################################
    y_norm_fun = None
    # Target coly processing, Normalization process  , customize by model
    log("y_norm_fun preprocess_pars")
    y_norm_fun = pars.get('y_norm_fun', None)
    if y_norm_fun is not None:
        df[coly] = df[coly].apply(lambda x: y_norm_fun(x))
        # save(y_norm_fun, f'{path_pipeline_export}/y_norm.pkl' )

    if pars.get('path_features_store', None) is not None:
        path_features_store = pars['path_features_store']
        save_features(df[coly], 'dfy', path_features_store)

    return df,col


def pd_colnum(df, col, pars):
    colnum = col
    for x in colnum:
        df[x] = df[x].astype("float32")
    log(df.dtypes)


def pd_colnum_fill_na_median(df, col, pars):
    for quant_col in col:
        df[quant_col].fillna((df[quant_col].median()), inplace=True)


def pd_colnum_normalize(df: pd.DataFrame, col: list=None, pars: dict=None):
    log("### colnum normalize  ###############################################################")
    from util_feature import pd_colnum_normalize
    colnum = col

    pars = { 'pipe_list': [ {'name': 'fillna', 'naval' : 0.0 }, {'name': 'minmax'} ]}
    dfnum_norm, colnum_norm = pd_colnum_normalize(df, colname=colnum,  pars=pars, suffix = "_norm",
                                                  return_val="dataframe,param")
    log(colnum_norm)
    if pars.get('path_features_store', None) is not None:
        path_features_store = pars['path_features_store']
        save_features(dfnum_norm, 'dfnum_norm', path_features_store)
    return dfnum_norm, colnum_norm


def pd_colnum_quantile_norm(df: pd.DataFrame, col: list=None, pars: dict=None):
  """
     colnum normalization by quantile
  """
  prefix  = "colnum_quantile_norm"
  df      = df[col]
  num_col = col

  ##### Grab previous computed params  ################################################
  pars2 = {}
  if  'path_pipeline' in pars :   #### Load existing column list
       colnum_quantile_norm = load( pars['path_pipeline']  +f'/{prefix}.pkl')
       model                = load( pars['path_pipeline']  +f'/{prefix}_model.pkl')
       pars2                = load( pars['path_pipeline']  +f'/{prefix}_pars.pkl')

  lower_bound_sparse = pars2.get('lower_bound_sparse', None)
  upper_bound_sparse = pars2.get('upper_bound_sparse', None)
  lower_bound        = pars2.get('lower_bound_sparse', None)
  upper_bound        = pars2.get('upper_bound_sparse', None)
  sparse_col         = pars2.get('colsparse', ['capital-gain', 'capital-loss'] )

  ####### Find IQR and implement to numericals and sparse columns seperately ##########
  Q1  = df.quantile(0.25)
  Q3  = df.quantile(0.75)
  IQR = Q3 - Q1

  for col in num_col:
    if col in sparse_col:
      df_nosparse = pd.DataFrame(df[df[col] != df[col].mode()[0]][col])

      if lower_bound_sparse is not None:
        pass

      elif df_nosparse[col].quantile(0.25) < df[col].mode()[0]: #Unexpected case
        lower_bound_sparse = df_nosparse[col].quantile(0.25)

      else:
        lower_bound_sparse = df[col].mode()[0]

      if upper_bound_sparse is not None:
        pass

      elif df_nosparse[col].quantile(0.75) < df[col].mode()[0]: #Unexpected case
        upper_bound_sparse = df[col].mode()[0]

      else:
        upper_bound_sparse = df_nosparse[col].quantile(0.75)

      n_outliers = len(df[(df[col] < lower_bound_sparse) | (df[col] > upper_bound_sparse)][col])

      if n_outliers > 0:
        df.loc[df[col] < lower_bound_sparse, col] = lower_bound_sparse * 0.75 #--> MAIN DF CHANGED
        df.loc[df[col] > upper_bound_sparse, col] = upper_bound_sparse * 1.25 # --> MAIN DF CHANGED

    else:
      if lower_bound is None or upper_bound is None :
         lower_bound = df[col].quantile(0.25) - 1.5 * IQR[col]
         upper_bound = df[col].quantile(0.75) + 1.5 * IQR[col]

      df[col] = np.where(df[col] > upper_bound, 1.25 * upper_bound, df[col])
      df[col] = np.where(df[col] < lower_bound, 0.75 * lower_bound, df[col])

  df.columns = [ t + "_qt_norm" for t in df.columns ]
  pars_new   = {'lower_bound' : lower_bound, 'upper_bound': upper_bound,
                'lower_bound_sparse' : lower_bound_sparse, 'upper_bound_sparse' : upper_bound_sparse  }
  dfnew    = df
  model    = None
  colnew   = list(df.columns)

  ##### Export ##############################################################################
  if 'path_features_store' in pars and 'path_pipeline_export' in pars:
      save_features(df,  prefix, pars['path_features_store'])
      save(colnew,     pars['path_pipeline_export']  + f"/{prefix}.pkl" )
      save(pars_new,   pars['path_pipeline_export']  + f"/{prefix}_pars.pkl" )
      save(model,      pars['path_pipeline_export']  + f"/{prefix}_model.pkl" )

  col_pars = {'prefix' : prefix, 'path': pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
  col_pars['cols_new'] = {
    prefix :  colnew  ### list
  }
  return dfnew,  col_pars


def pd_colnum_bin(df: pd.DataFrame, col: list=None, pars: dict=None):
    from util_feature import  pd_colnum_tocat

    path_pipeline  = pars.get('path_pipeline', False)
    colnum_binmap  = load(f'{path_pipeline}/colnum_binmap.pkl') if  path_pipeline else None
    log(colnum_binmap)

    colnum = col

    log("### colnum Map numerics to Category bin  ###########################################")
    dfnum_bin, colnum_binmap = pd_colnum_tocat(df, colname=colnum, colexclude=None, colbinmap=colnum_binmap,
                                               bins=10, suffix="_bin", method="uniform",
                                               return_val="dataframe,param")
    log(colnum_binmap)
    ### Renaming colunm_bin with suffix
    colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
    log(colnum_bin)

    if 'path_features_store' in pars:
        scol = "_".join(col[:5])
        save_features(dfnum_bin, 'colnum_bin' + "-" + scol, pars['path_features_store'])
        save(colnum_binmap,  pars['path_pipeline_export'] + "/colnum_binmap.pkl" )
        save(colnum_bin,     pars['path_pipeline_export'] + "/colnum_bin.pkl" )


    col_pars = {}
    col_pars['colnumbin_map'] = colnum_binmap
    col_pars['cols_new'] = {
     'colnum'     :  col ,    ###list
     'colnum_bin' :  colnum_bin       ### list
    }
    return dfnum_bin, col_pars


def pd_colnum_binto_onehot(df: pd.DataFrame, col: list=None, pars: dict=None):
    assert isinstance(col, list) and isinstance(df, pd.DataFrame)

    dfnum_bin     = df[col]
    colnum_bin    = col

    path_pipeline = pars.get('path_pipeline', False)
    colnum_onehot = load(f'{path_pipeline}/colnum_onehot.pkl') if  path_pipeline else None


    log("###### colnum bin to One Hot  #################################################")
    from util_feature import  pd_col_to_onehot
    dfnum_hot, colnum_onehot = pd_col_to_onehot(dfnum_bin[colnum_bin], colname=colnum_bin,
                                                colonehot=colnum_onehot, return_val="dataframe,param")
    log(colnum_onehot)

    if 'path_features_store' in pars :
        save_features(dfnum_hot, 'colnum_onehot', pars['path_features_store'])
        save(colnum_onehot,  pars['path_pipeline_export'] + "/colnum_onehot.pkl" )

    col_pars = {}
    col_pars['colnum_onehot'] = colnum_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colnum_onehot' :  colnum_onehot       ### list
    }
    return dfnum_hot, col_pars



def pd_colcat_to_onehot(df: pd.DataFrame, col: list=None, pars: dict=None):
    """

    """
    log("#### colcat to onehot")
    col         = [col]  if isinstance(col, str) else col
    if len(col)==1:
        colnew       = [col[0] + "_onehot"]
        df[ colnew ] =  df[col]
        col_pars     = {}
        col_pars['colcat_onehot'] = colnew
        col_pars['cols_new'] = {
         # 'colnum'        :  col ,    ###list
         'colcat_onehot'   :  colnew      ### list
        }
        return df[colnew], col_pars

    colcat_onehot = None
    if  'path_pipeline' in pars :
       colcat_onehot = load( pars['path_pipeline'] + '/colcat_onehot.pkl')

    ######################################################################################
    colcat = col
    dfcat_hot, colcat_onehot = util_feature.pd_col_to_onehot(df[colcat], colname=colcat,
                                                colonehot=colcat_onehot, return_val="dataframe,param")
    log(dfcat_hot[colcat_onehot].head(5))

    ######################################################################################
    if 'path_features_store' in pars :
        save_features(dfcat_hot, 'colcat_onehot', pars['path_features_store'])
        save(colcat_onehot,  pars['path_pipeline_export'] + "/colcat_onehot.pkl" )
        save(colcat,         pars['path_pipeline_export'] + "/colcat.pkl" )

    col_pars = {}
    col_pars['colcat_onehot'] = colcat_onehot
    col_pars['cols_new'] = {
     # 'colnum'        :  col ,    ###list
     'colcat_onehot' :  colcat_onehot       ### list
    }

    print("ok ------------")
    return dfcat_hot, col_pars



def pd_colcat_bin(df: pd.DataFrame, col: list=None, pars: dict=None):
    # dfbum_bin = df[col]
    path_pipeline  = pars.get('path_pipeline', False)
    colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl') if  path_pipeline else None
    colcat         = [col]  if isinstance(col, str) else col

    log("#### Colcat to integer encoding ")
    dfcat_bin, colcat_bin_map = util_feature.pd_colcat_toint(df[colcat], colname=colcat,
                                                            colcat_map=  colcat_bin_map ,
                                                            suffix="_int")
    colcat_bin = list(dfcat_bin.columns)
    ##### Colcat processing   ################################################################
    colcat_map = util_feature.pd_colcat_mapping(df, colcat)
    log(df[colcat].dtypes, colcat_map)


    if 'path_features_store' in pars :
       save_features(dfcat_bin, 'dfcat_bin', pars['path_features_store'])
       save(colcat_bin_map,  pars['path_pipeline_export'] + "/colcat_bin_map.pkl" )
       save(colcat_bin,      pars['path_pipeline_export'] + "/colcat_bin.pkl" )


    col_pars = {}
    col_pars['colcat_bin_map'] = colcat_bin_map
    col_pars['cols_new'] = {
     'colcat'     :  col ,    ###list
     'colcat_bin' :  colcat_bin       ### list
    }

    return dfcat_bin, col_pars



def pd_colcross(df: pd.DataFrame, col: list=None, pars: dict=None):
    """
     cross_feature_new =  feat1 X feat2  (pair feature)

    """
    log("#####  Cross Features From OneHot Features   ######################################")
    prefix = 'colcross_onehot'

    # params_check(pars,  [('dfcat_hot', pd.DataFrame), 'colid',   ])
    from util_feature import pd_feature_generate_cross

    dfcat_hot = pars['dfcat_hot']
    colid     = pars['colid']

    try :
       dfnum_hot = pars['dfnum_hot']
       df_onehot = dfcat_hot.join(dfnum_hot, on=colid, how='left')
    except :
       df_onehot = copy.deepcopy(dfcat_hot)

    colcross_single = pars['colcross_single']
    pars_model      = { 'pct_threshold' :0.02,  'm_combination': 2 }
    if  'path_pipeline' in pars :   #### Load existing column list
       colcross_single = load( pars['path_pipeline']  + f'/{prefix}_select.pkl')
       # pars_model      = load( pars['path_pipeline']  + f'/{prefix}_pars.pkl')

    colcross_single_onehot_select = []  ## Select existing columns
    for t in list(df_onehot.columns):
       for c1 in colcross_single:
           if c1 in t:
               colcross_single_onehot_select.append(t)

    df_onehot = df_onehot[colcross_single_onehot_select ]
    dfcross_hot, colcross_pair = pd_feature_generate_cross(df_onehot, colcross_single_onehot_select,
                                                           **pars_model)
    log(dfcross_hot.head(2).T)
    colcross_pair_onehot = list(dfcross_hot.columns)

    model = None
    ##############################################################################
    if 'path_features_store' in pars:
        save_features(dfcross_hot, 'colcross_onehot', pars['path_features_store'])
        save(colcross_single_onehot_select, pars['path_pipeline_export'] + f'/{prefix}_select.pkl')
        save(colcross_pair,                 pars['path_pipeline_export'] + f'/{prefix}_stats.pkl')
        save(colcross_pair_onehot,          pars['path_pipeline_export'] + f'/{prefix}_pair.pkl')
        save(model,                         pars['path_pipeline_export'] + f'/{prefix}_pars.pkl')

    col_pars = {'model': model, 'stats' : colcross_pair }
    col_pars['cols_new'] = {
     # 'colcross_single'     :  col ,    ###list
     'colcross_pair' :  colcross_pair_onehot       ### list
    }
    return dfcross_hot, col_pars



def pd_coldate(df: pd.DataFrame, col: list=None, pars: dict=None):
    log("##### Coldate processing   ##########################################")
    from utils import util_date
    coldate = col
    dfdate  = None
    for coldate_i in coldate :
        dfdate_i = util_date.pd_datestring_split( df[[coldate_i]] , coldate_i, fmt="auto", return_val= "split" )
        dfdate   = pd.concat((dfdate, dfdate_i),axis=1)  if dfdate is not None else dfdate_i
        # if 'path_features_store' in pars :
        #    path_features_store = pars['path_features_store']
        #    #save_features(dfdate_i, 'dfdate_' + coldate_i, path_features_store)

    if 'path_features_store' in pars :
        save_features(dfdate, 'dfdate', pars['path_features_store'])

    col_pars = {}
    col_pars['cols_new'] = {
        # 'colcross_single'     :  col ,    ###list
        'dfdate': list(dfdate.columns)  ### list
    }
    return dfdate, col_pars


def pd_colcat_encoder_generic(df: pd.DataFrame, col: list=None, pars: dict=None):
    """
        Create a Class or decorator
        https://pypi.org/project/category-encoders/
        encoder = ce.BackwardDifferenceEncoder(cols=[...])
        encoder = ce.BaseNEncoder(cols=[...])
        encoder = ce.BinaryEncoder(cols=[...])
        encoder = ce.CatBoostEncoder(cols=[...])
        encoder = ce.CountEncoder(cols=[...])
        encoder = ce.GLMMEncoder(cols=[...])
        encoder = ce.HashingEncoder(cols=[...])
        encoder = ce.HelmertEncoder(cols=[...])
        encoder = ce.JamesSteinEncoder(cols=[...])
        encoder = ce.LeaveOneOutEncoder(cols=[...])
        encoder = ce.MEstimateEncoder(cols=[...])
        encoder = ce.OneHotEncoder(cols=[...])
        encoder = ce.OrdinalEncoder(cols=[...])
        encoder = ce.SumEncoder(cols=[...])
        encoder = ce.PolynomialEncoder(cols=[...])
        encoder = ce.TargetEncoder(cols=[...])
        encoder = ce.WOEEncoder(cols=[...])
    """
    prefix     = "colcat_encoder_generic"
    pars_model = None
    if 'path_pipeline' in  pars  :   ### Load during Inference
       colcat_encoder = load( pars['path_pipeline'] + f"/{prefix}.pkl" )
       pars_model     = load( pars['path_pipeline'] + f"/{prefix}_pars.pkl" )
       #model         = load( pars['path_pipeline'] + f"/{prefix}_model.pkl" )

    ####### Custom Code ###############################################################
    from category_encoders import HashingEncoder, WOEEncoder
    pars_model         = pars.get('model_pars', {})  if pars_model is None else pars_model
    pars_model['cols'] = col
    model_name         = pars.get('model_name', 'HashingEncoder')

    model_class        = { 'HashingEncoder' : HashingEncoder  }[model_name]
    model              = model_class(**pars_model)
    dfcat_encoder      = model.fit_transform(df[col])

    dfcat_encoder.columns = [t + "_cod" for t in dfcat_encoder.columns ]
    colcat_encoder        = list(dfcat_encoder.columns)


    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfcat_encoder, 'dfcat_encoder', pars['path_features_store'])
       save(model,           pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(pars_model,      pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       save(colcat_encoder,  pars['path_pipeline_export'] + f"/{prefix}.pkl" )

    col_pars = { 'prefix' : prefix,  'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
     'colcat_encoder_generic' :  colcat_encoder  ### list
    }
    return dfcat_encoder, col_pars



def pd_colcat_minhash(df: pd.DataFrame, col: list=None, pars: dict=None):
    """
       MinHash Algo for category
       https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087

    """
    prefix = 'colcat_minhash'
    colcat              = col

    pars_minhash = {'n_component' : [4, 2], 'model_pretrain_dict' : None,}
    if 'path_pipeline_export' in pars :
        try :
            pars_minhash = load( pars['path_pipeline_export'] + '/colcat_minhash_pars.pkl')
        except : pass

    log("#### Colcat to Hash encoding #############################################")
    from utils import util_text
    dfcat_bin, col_hash_model= util_text.pd_coltext_minhash(df[colcat], colcat,
                                                            return_val="dataframe,param", **pars_minhash )
    colcat_minhash = list(dfcat_bin.columns)
    log(col_hash_model)

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfcat_bin, prefix, pars['path_features_store'])
       save(colcat_minhash, pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_minhash,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       save(col_hash_model, pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )

    col_pars = {}
    col_pars['col_hash_model'] = col_hash_model
    col_pars['cols_new'] = {
     'colcat_minhash' :  colcat_minhash  ### list
    }
    return dfcat_bin, col_pars



def os_convert_topython_code(txt):
    # from sympy import sympify
    # converter = {
    #     'sub': lambda x, y: x - y,
    #     'div': lambda x, y: x / y,
    #     'mul': lambda x, y: x * y,
    #     'add': lambda x, y: x + y,
    #     'neg': lambda x: -x,
    #     'pow': lambda x, y: x ** y
    # }
    # formula = sympify( txt, locals=converter)
    # print(formula)
    pass


def save_json(js, pfile, mode='a'):
    import  json
    with open(pfile, mode=mode) as fp :
        json.dump(js, fp)


def pd_col_genetic_transform(df: pd.DataFrame, col: list=None, pars: dict=None):
    """
        Find Symbolic formulae for faeture engineering

    """
    prefix = 'col_genetic'
    ######################################################################################
    from gplearn.genetic import SymbolicTransformer
    from gplearn.functions import make_function
    import random

    colX          = col # [col_ for col_ in col if col_ not in coly]
    train_X       = df[colX].fillna(method='ffill')
    feature_name_ = colX

    def squaree(x):  return x * x
    square_ = make_function(function=squaree, name='square_', arity=1)

    function_set = pars.get('function_set',
                            ['add', 'sub', 'mul', 'div',  'sqrt', 'log', 'abs', 'neg', 'inv','tan', square_])
    pars_genetic = pars.get('pars_genetic',
                             {'generations': 5, 'population_size': 10,  ### Higher than nb_features
                              'metric': 'spearman',
                              'tournament_size': 20, 'stopping_criteria': 1.0, 'const_range': (-1., 1.),
                              'p_crossover': 0.9, 'p_subtree_mutation': 0.01, 'p_hoist_mutation': 0.01,
                              'p_point_mutation': 0.01, 'p_point_replace': 0.05,
                              'parsimony_coefficient' : 0.005,   ####   0.00005 Control Complexity
                              'max_samples' : 0.9, 'verbose' : 1,

                              #'n_components'      ### Control number of outtput features  : n_components
                              'random_state' :0, 'n_jobs' : 4,
                              })

    if 'path_pipeline' in pars :   #### Inference time
        gp   = load(pars['path_pipeline'] + f"/{prefix}_model.pkl" )
        pars = load(pars['path_pipeline'] + f"/{prefix}_pars.pkl"  )
    else :     ### Training time
        coly     = pars['coly']
        train_y  = pars['dfy']
        gp = SymbolicTransformer(hall_of_fame  = train_X.shape[1] + 1,  ### Buggy
                                 n_components  = pars_genetic.get('n_components', train_X.shape[1] ),
                                 feature_names = feature_name_,
                                 function_set  = function_set,
                                 **pars_genetic)
        gp.fit(train_X, train_y)

    ##### Transform Data  #########################################
    df_genetic = gp.transform(train_X)
    tag = random.randint(0,10)   #### UNIQUE TAG
    col_genetic  = [ f"gen_{tag}_{i}" for i in range(df_genetic.shape[1])]
    df_genetic   = pd.DataFrame(df_genetic, columns= col_genetic, index = train_X.index )
    df_genetic.index = train_X.index
    pars_gen_all = {'pars_genetic'  : pars_genetic , 'function_set' : function_set }

    ##### Formulae Exrraction #####################################
    formula   = str(gp).replace("[","").replace("]","")
    flist     = formula.split(",\n")
    form_dict = {  x: flist[i]  for i,x in enumerate(col_genetic) }
    pars_gen_all['formulae_dict'] = form_dict
    log("########## Formulae ", form_dict)
    # col_pars['map_dict'] = dict(zip(train_X.columns.to_list(), feature_name_))

    col_new = col_genetic

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df_genetic, 'df_genetic', pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col_genetic,    pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_gen_all,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       # save(form_dict,      pars['path_pipeline_export'] + f"/{prefix}_formula.pkl")
       save_json(form_dict, pars['path_pipeline_export'] + f"/{prefix}_formula.json")   ### Human readable


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ### list
    }
    return df_genetic, col_pars









######################################################################################
def test():
    from util_feature import test_get_classification_data
    dfX, dfy = test_get_classification_data()
    cols     = list(dfX.columsn)
    ll       = [ ('pd_colnum_bin', {}  )
               ]

    for fname, pars in ll :
        try :
           myfun = globals()[fname]
           res   = myfun(dfX, cols, pars)
           log( f"Success, {fname}, {pars}, {e}")
        except Exception as e :
            log( f"Failed, {fname}, {pars}, {e}")




if __name__ == "__main__":
    import fire
    fire.Fire()
