import pandas as pd
import random, os, sys
import numpy as np
from source.prepro_tseries import *


#### Add path for python import  #######################################
path_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
print("path_repo_root", path_repo_root)
sys.path.append( path_repo_root)
########################################################################

folder     = 'raw/'
df         = pd.read_csv(folder+'raw_data.csv', delimiter=',')


df_date, cols_info = pd_ts_date(df, cols=["Date"], pars = {"col_add" : ["day"]})
df_rolling, cols_rolling = pd_ts_rolling(df, cols = ["Date", "Adj Close"], pars = {"col_groupby" : ["Date"], "col_stat" : "Adj Close"})
df_lag, cols_lag = pd_ts_lag(df, cols = ["Date", "Adj Close"], pars = {"col_groupby" : ["Date"], "col_stat" : "Adj Close"})
pd.concat([df_lag, df_rolling, df_date], axis = 1)
tsfresh_features, cols_tsfresh = pd_ts_tsfresh_features(df, ["Adj Close"], {})
