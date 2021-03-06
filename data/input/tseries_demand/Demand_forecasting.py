#import dependencies

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set()
import plotly.offline as py
import plotly.graph_objs as go
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load the Train and Test data
def load_data(datapath):
    data = pd.read_csv(datapath)
    # Dimensions
    print('Shape:', data.shape)
    return data
    
# change the path accordingly    
train_df = load_data('./train/features.csv')
test_df = load_data('./test/features.csv')


# Preprocess functions from prepro_tseries
def pd_ts_date(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """ DataFrame with date column"""
    df      = df[cols]
    coldate = [cols] if isinstance(cols, str) else cols
    print(coldate)
    col_add = pars.get('col_add', ['day', ',month'])
    print(col_add)
    dfdate  =  None
    df2     = pd.DataFrame()
    for coli in coldate:
        df2[coli]               = pd.to_datetime(df[coli], errors='coerce')
        if 'day'  in col_add      : df2[coli + '_day']      = df2[coli].dt.day
        if 'month'  in col_add    : df2[coli + '_month']    = df2[coli].dt.month
        if 'year'  in col_add     : df2[coli + '_year']     = df2[coli].dt.year
        if 'hour'  in col_add     : df2[coli + '_hour']     = df2[coli].dt.hour
        if 'minute'  in col_add   : df2[coli + '_minute']   = df2[coli].dt.minute
        if 'weekday'  in col_add  : df2[coli + '_weekday']  = df2[coli].dt.weekday
        if 'dayyear'  in col_add  : df2[coli + '_dayyear']  = df2[coli].dt.dayofyear
        if 'weekyear'  in col_add : df2[coli + '_weekyear'] = df2[coli].dt.weekofyear
        dfdate = pd.concat((dfdate, df2 )) if dfdate is not None else df2
        del dfdate[coli]  ### delete col

    ##### output  ##########################################
    col_pars = {}
    col_pars['cols_new'] = {
        # 'colcross_single'     :  col ,    ###list
        'dfdate': list(dfdate.columns)  ### list
    }
    return dfdate, col_pars



def pd_ts_rolling(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """
      Rolling statistics

    """
    cat_cols     = []
    col_new      = []
    id_cols      = []
    colgroup     = pars.get('col_groupby', ['id'])
    colstat      = pars['col_stat']
    lag_list     = pars.get('lag_list', [7, 14, 30, 60, 180])
    len_shift    = pars.get('len_shift', 28)

    len_shift_list   = pars.get('len_shift_list' , [1,7,14])
    len_window_list  = pars.get('len_window_list', [7, 14, 30, 60])


    for i in lag_list:
        print('Rolling period:', i)
        df['rolling_mean_' + str(i)] = df.groupby(colgroup)[colstat].transform(
            lambda x: x.shift(len_shift).rolling(i).mean())

        df['rolling_std_' + str(i)] = df.groupby(colgroup)[colstat].transform(
            lambda x: x.shift(len_shift).rolling(i).std())

        col_new.append('rolling_mean_' + str(i))
        col_new.append('rolling_std_' + str(i))


    # Rollings with sliding shift
    for len_shift in len_shift_list:
        print('Shifting period:', len_shift)
        for len_window in len_window_list:
            col_name = f'rolling_mean_tmp_{len_shift}_{len_window}'
            df[col_name] = df.groupby(colgroup)[colstat].transform(
                lambda x: x.shift(len_shift).rolling(len_window).mean())
            col_new.append(col_name)

    for col_name in id_cols:
        col_new.append(col_name)

    return df[col_new], cat_cols



 
def preprocessing_data(df):
    # df['date'] = pd.to_datetime(df['date'])
    colid = 'date'
    df = df.set_index(colid)

    #### time features
    df1, col1 = pd_ts_date(df, cols=['date'], pars={'col_add':['day', 'month', 'year', 'weekday']})
    dfall = pd.concat([df, df2], axis=1)
    
    #### lag features, rolling window features
    df2, col2 = pd_ts_rolling(df, 
                              cols= ['date', 'item', 'store', 'sales'], 
                              pars= {'col_groupby' : ['store','item'],
                                     'col_stat':     'sales', 'lag_list': [7, 30]})    
    dfall = pd.concat([dfall, df2], axis=1)



    ##### 
    col = [i for i in dfall.columns if i not in ['date','id']]
    y   = 'sales'

    return dfall, col, y




df, col, y = preprocessing_data(train_df)


########################################################################################
# split into train and validation and test data
train_x, test_x, train_y, test_y = train_test_split(df[col],df[y], test_size=0.3, random_state=2018)
val_x, test_x, val_y, test_y = train_test_split(test_x[col],test_y, test_size=0.33, random_state=2018)

# Train function
def train_model(train_x,train_y,test_x,test_y,col):
    params = {
        'nthread': 10,
        'max_depth': 5,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mape', # this is abs(a-e)/max(1,a)
        'num_leaves': 64,
        'learning_rate': 0.2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 3.097758978478437,
        'lambda_l2': 2.9482537987198496,
        'verbose': 1,
        'min_child_weight': 6.996211413900573,
        'min_split_gain': 0.037310344962162616,
        }
    
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(test_x,test_y)
    model = lgb.train(params, lgb_train, 30, valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=50, verbose_eval=50)
    return model

model = train_model(train_x,train_y,test_x,test_y,col)

# Predict
predicted = model.predict(test_x[col])
# save predicted sales to csv file
test_x['sales'] = test_y
test_x['Predicted_sales'] = predicted
test_x['error'] = test_x['sales']- test_x['Predicted_sales']
test_x.to_csv('./Predicted_Sales.csv')
