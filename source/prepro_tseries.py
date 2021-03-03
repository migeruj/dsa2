# -*- coding: utf-8 -*-
"""
 Time Series preprocessing tools :

   Transform time series  (zt) INTO  supervised problem  yi = F(Xi)


  coldate                        : Parse the date and split date into columns
  groupby features               : Using data within a group ;  For each date,  aggregate col. over the groupby
  smooth, autoregressive feature : Using Past data :   For each date, a Fixed column,  aggregate over past time window.

  Example: Daily sales per item_id, shop_id, zone_id

       date --> pd_coldate

       groupby(shop_id) ---> per each date,  mean, max, min ...

       groupby(zone_id) ---> per each date,  mean, max, min ...

       item_id sales -->  per each date, Moving Average, Min, Max over 1month, ...

"""
import warnings, os, sys, re
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, copy
import pdb

####################################################################################################
#### Add path for python import
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/")

#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

DEBUG= True


####################################################################################################
from util_feature import (load, save_list, load_function_uri,  save,
                          save_features, load_features)


def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    print(*s, fluhs=True)


def logd(*s, n=0, m=0):
    if DEBUG :
        sspace = "#" * n
        sjump = "\n" * m
        print(*s, fluhs=True)


####################################################################################################
try:
    from tsfresh import extract_relevant_features, extract_features
    from tsfresh.utilities.dataframe_functions import roll_time_series
    from deltapy import transform, interact, mapper, extract
    import pandasvault, tsfel, datetime as dt

except:
    os.system(" pip install  tsfresh pandasvault tsfel")
    os.system(" pip install deltapy ")

    from tsfresh import extract_relevant_features, extract_features
    from tsfresh.utilities.dataframe_functions import roll_time_series
    from deltapy import transform, interact, mapper, extract
    import pandasvault, datetime as dt





###########################################################################################
###########################################################################################
def pd_ts_date(df: pd.DataFrame, cols: list=None, pars: dict=None):

    df      = df[cols]
    coldate = [cols] if isinstance(cols, str) else cols
    col_add = pars.get('col_add', ['day', ',month'])

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




def pd_ts_groupby(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """
        Generates features by groupBy over category
        groupby(shop_id) ---> per each date,  mean, max, min ...
        groupby(zone_id) ---> per each date,  mean, max, min ...

        groupby(key_lis).agg( col_stat )

    """
    colgroup  = pars.get('colgroupby')    #### list of list of columns for aggregation
    colstat   = pars.get('colstat')     ####  column whwere : sales, amount,
    calc_list = pars.get('calc_list', {'mean'})   ### what kind of stats

    colgroup_merge = [ colj for colgroupi in colgroup for colj in colgroupi ]   #### Flatten
    colgroup_merge = list(set(colgroup_merge))
    dfall          = df[colgroup_merge].drop_duplicates(  colgroup_merge )

    for colgroupi in colgroup:
        df1   = df.groupby(colgroupi).agg({coli: set(calc_list) for coli in colstat})
        dfall = dfall.join(df1, on=colgroupi, how='left')

    return dfall


def pd_ts_onehot(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """
       category to oneHot (ie week, weekday, shop, ..)
    :param df:
    :param col:
    :param pars:
    :return:
    """
    df_onehot = pd.get_dummies(df[cols])
    # df = pd.concat([df, dummy_cols], axis=1)
    return df_onehot



def pd_ts_autoregressive(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """
        Using past data for same column ; Rolling
        item_id sales -->  per each date, Moving Average, Min, Max over 1month, ...
        shop_id sales -->  per each date, Moving Average, Min, Max over 1month, ...
    """
    pass



def pd_ts_rolling(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """
      Rolling statistics

    """
    cat_cols     = []
    col_new      = []
    colgroup     = pars.get('col_groupby', ['id'])
    colstat      = pars['col_stat']
    lag_list     = pars.get('lag_list', [7, 14, 30, 60, 180])
    id_cols      = []

    len_shift = 28
    for i in lag_list:
        print('Rolling period:', i)
        df['rolling_mean_' + str(i)] = df.groupby(colgroup)[colstat].transform(
            lambda x: x.shift(len_shift).rolling(i).mean())

        df['rolling_std_' + str(i)] = df.groupby(colgroup)[colstat].transform(
            lambda x: x.shift(len_shift).rolling(i).std())

        col_new.append('rolling_mean_' + str(i))
        col_new.append('rolling_std_' + str(i))

    # Rollings
    # with sliding shift
    for len_shift in [1, 7, 14]:
        print('Shifting period:', len_shift)
        for len_window in [7, 14, 30, 60]:
            col_name = 'rolling_mean_tmp_' + str(len_shift) + '_' + str(len_window)
            df[col_name] = df.groupby(colgroup)[colstat].transform(
                lambda x: x.shift(len_shift).rolling(len_window).mean())
            col_new.append(col_name)

    for col_name in id_cols:
        col_new.append(col_name)

    return df[col_new], cat_cols


def pd_ts_lag(df: pd.DataFrame, cols: list=None, pars: dict=None):
    col_new = []
    cat_cols     = []
    id_cols = []
    colgroupby = pars.get('col_groupby', ['id'])
    colstat      = pars['col_stat']

    lag_days = [col for col in range(28, 28 + 15)]
    for lag_day in lag_days:
        col_new.append('lag_' + str(lag_day))
        df['lag_' + str(lag_day)] = df.groupby(colgroupby)[colstat].transform(lambda x: x.shift(lag_day))

    for col_name in id_cols:
        col_new.append(col_name)

    return df[col_new], cat_cols



def pd_ts_difference(df: pd.DataFrame, cols: list=None, pars: dict=None):
    lag  = pars.get('lag', 1)
    df = df[cols]
    for col in cols :
       df[col] = df[col].diff(lag=lag)

    return df



# def pd_ts_tsfresh_features(df: pd.DataFrame, cols: list=None, pars: dict=None):
    # """
    #
    # :param df:
    # :param cols:
    # :return:
    # """
    #
    # df_cols                 = df.columns.tolist()
    # single_row_df_T         = df[cols].T
    # single_row_df_T["time"] = range(0, len(single_row_df_T.index))
    # single_row_df_T["id"]   = range(0, len(single_row_df_T.index))
    # single_row_df_T.rename(columns={single_row_df_T.columns[0]: "val"}, inplace=True)
    #
    # X_feat = extract_features(single_row_df_T, column_id='id', column_sort='time')
    #
    # feat_col_names = X_feat.columns.tolist()
    # feat_col_names_mapping = {}
    # for feat_col_name in feat_col_names:
    #     feat_col_names_mapping[feat_col_name] = feat_col_name.replace('"', '').replace(',', '')
    #
    # X_feat = X_feat.rename(columns=feat_col_names_mapping)
    # X_feat_T = X_feat.T
    #
    # for col in cols:
    #     X_feat_T[col] = np.repeat(df[col].tolist()[0], len(X_feat_T.index))
    # # X_feat_T["item_id"] = np.repeat(df["item_id"].tolist()[0], len(X_feat_T.index))
    # # X_feat_T["id"] = np.repeat(df["id"].tolist()[0], len(X_feat_T.index))
    # # X_feat_T["cat_id"] = np.repeat(df["cat_id"].tolist()[0], len(X_feat_T.index))
    # # X_feat_T["dept_id"] = np.repeat(df["dept_id"].tolist()[0], len(X_feat_T.index))
    # # X_feat_T["store_id"] = np.repeat(df["store_id"].tolist()[0], len(X_feat_T.index))
    # # X_feat_T["state_id"] = np.repeat(df["state_id"].tolist()[0], len(X_feat_T.index))
    # X_feat_T["variable"] = X_feat_T.index
    #
    # df["variable"] = pd.Series(["demand"])
    # X_feat_T = X_feat_T.append(df, ignore_index=True)
    # pdb.set_trace()
    # return X_feat_T.set_index(cols + ['variable']).rename_axis(['day'], axis=1).stack().unstack(
    #     'variable').reset_index()


def pd_ts_tsfresh_features(df: pd.DataFrame, cols: list=None, pars: dict=None):
    single_row_df         = df[cols]
    single_row_df["time"] = range(0, len(single_row_df.index))
    id_col = pars.get("id_col", "id")
    if not "id_col" in pars.keys():
        single_row_df["id"] = 1
    X_feat = extract_features(single_row_df, column_id=id_col, column_sort='time')
    return X_feat,  X_feat.columns.to_list()


def pd_ts_deltapy_generic(df: pd.DataFrame, cols: list=None, pars: dict=None ):
    """
       { 'name': 'deltapy.transform::robust_scaler',                 'pars': {'drop':["Close_1"]} },

    """
    ###### Custom code ################################################################
    model_name = pars['name']
    model_pars = pars.get('pars', {})

    dfin = df[cols]
    dfin = dfin.fillna(method='ffill')

    if 'a_chi' in model_name:
        # Normalize the input for the chi
        dfin = (dfin - dfin.min()) / (dfin.max() - dfin.min())

    ##### Transform Data  ############################################################
    model  = load_function_uri(model_name)
    df_out = model(dfin, **model_pars)

    if 'extract' in model_name:
        # Extract only returns one value, so no columns to loop over.
        col_out = "0_" + model_name

    else:
        model_name2    = model_name.replace("::", "-")
        col_out        = [coli + "_" + model_name2 for coli in df_out.columns]
        df_out.columns = col_out
        df_out.index   = df.index

    return df_out


def pd_ts_deltapy2(df: pd.DataFrame, cols: list=None, pars: dict=None ):
    """
       Delta py
       pars : {  'name' :  "robust_scaler",
                 'pars'  :  {}
       }
    """
    prefix = 'colts_deltapy'

    ###### Custom code ################################################################
    dfin = df.fillna(method='ffill')
    model_name = pars['name']
    model_pars = pars.get('pars', {})

    if 'path_pipeline' in pars:  #### Prediction time
        model = load(pars['path_pipeline'] + f"/{prefix}_model.pkl")
        pars = load(pars['path_pipeline'] + f"/{prefix}_pars.pkl")

    else:  ### Training time  : Dynamic function load
        from util_feature import load_function_uri
        ##### transform.robust_scaler(df, drop=["Close_1"])
        model = load_function_uri(model_name)

    ##### Transform Data  ############################################################
    df_out = model(dfin, **model_pars)

    # Extract only returns one value, so no columns to loop over.
    model_name2 = model_name.replace("::", "-")
    if 'extract' in model_name:
        col_out = "0_" + model_name
    else:
        col_out = [coli + "_" + model_name for coli in df_out.columns]
        df_out.columns = col_out
        df_out.index = df_out.index
    col_new = col_out

    ###### Export #####################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
        save_features(df_out, 'df_' + prefix, pars['path_features_store'])
        save(model, pars['path_pipeline_export'] + f"/{prefix}_model.pkl")
        save(col_new, pars['path_pipeline_export'] + f"/{prefix}.pkl")
        save(pars, pars['path_pipeline_export'] + f"/{prefix}_pars.pkl")

    col_pars = {'prefix': prefix, 'path': pars.get('path_pipeline_export', pars.get('path_pipeline', None))}
    col_pars['cols_new'] = {
        prefix: col_new  ### list of columns
    }
    return df_out, col_pars









########################################################################################################################
########################################################################################################################

def pd_ts_custom(df: pd.DataFrame, cols: list=None, pars: dict=None):
    """   Generic template for feature generation
       'colnum' : ['sales1' 'units' ]
      'pars_function_list' :  [
       { 'name': 'deltapy.transform::robust_scaler',                 'pars': {'drop':["Close_1"]} },
       { 'name': 'deltapy.transform::standard_scaler',               'pars': {'drop':["Close_1"]} },
       ]e

    :param df:
    :param col:
    :param pars:
    :return:
    """

    df      = df[col]
    coldate = pars['coldate']
    colnum  = pars['colnum']
    colcat  = pars['colcat']

    colgroups = pars['colgroup']
    colgstat  = pars['colstat']

    log("### Only dates")
    df1 = pd_ts_date(df, coldate, pars)
    coldate1 = list(df1.columns)

    log("### Initial features")
    df1 = df1.join(df, on=coldate, how='left')

    log("### Groupby features")
    df2 = pd_ts_groupby(df, col, pars)
    df1 = df1.join(df2, on=coldate, how='left')

    log("### Numerical features")
    colnum2 = list(df2.columns) + colnum
    df1     = df1.set_index(coldate1)

    log("### Deltapy features")
    for pars_function_dict_i in pars.get('pars_function_list', []):
        dfi = pd_ts_deltapy_generic(df1, col=colnum2, pars=pars_function_dict_i)
        df1 = df1.join(dfi, on=coldate, how='left')

    return df1




########################################################################################################################
########################################################################################################################
def test_get_sampledata(url="https://github.com/firmai/random-assets-two/raw/master/numpy/tsla.csv"):
    df = pd.read_csv(url)
    df["Close_1"] = df["Close"].shift(-1)
    with pd.option_context('mode.use_inf_as_na', True):
        df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df         = df.set_index("Date")
    return df


def test_deltapy_get_method(df):
    prepro_list = [
        {'name': 'deltapy.transform::robust_scaler', 'pars': {'drop': ["Close_1"]}},
        {'name': 'deltapy.transform::standard_scaler', 'pars': {'drop': ["Close_1"]}},
        {'name': 'deltapy.transform::fast_fracdiff', 'pars': {'cols': ["Close", "Open"], 'd': 0.5}},
        {'name': 'deltapy.transform::operations', 'pars': {'features': ["Close"]}},
        {'name': 'deltapy.transform::triple_exponential_smoothing',
         'pars': {'cols': ["Close"], 'slen': 12, 'alpha': .2, 'beta': .5, 'gamma': .5, 'n_preds': 0}},
        {'name': 'deltapy.transform::naive_dec', 'pars': {'columns': ["Close", "Open"]}},
        {'name': 'deltapy.transform::bkb', 'pars': {'cols': ["Close"]}},
        {'name': 'deltapy.transform::butter_lowpass_filter', 'pars': {'cols': ["Close"], 'cutoff': 4}},
        {'name': 'deltapy.transform::instantaneous_phases', 'pars': {'cols': ["Close"]}},
        {'name': 'deltapy.transform::kalman_feat', 'pars': {'cols': ["Close"]}},
        {'name': 'deltapy.transform::perd_feat', 'pars': {'cols': ["Close"]}},
        {'name': 'deltapy.transform::fft_feat', 'pars': {'cols': ["Close"]}},
        {'name': 'deltapy.transform::harmonicradar_cw', 'pars': {'cols': ["Close"], 'fs': 0.3, 'fc': 0.2}},
        {'name': 'deltapy.transform::saw', 'pars': {'cols': ["Close", "Open"]}},
        {'name': 'deltapy.transform::multiple_rolling', 'pars': {'columns': ["Close"]}},
        {'name': 'deltapy.transform::multiple_lags', 'pars': {'columns': ["Close"], 'start': 1, 'end': 3}},

        {'name': 'deltapy.interact::lowess',
         'pars': {'cols': ["Open", "Volume"], 'y': df["Close"], 'f': 0.25, 'iter': 3}},

        {'name': 'deltapy.interact::autoregression', 'pars': {}},
        {'name': 'deltapy.interact::muldiv', 'pars': {'feature_list': ["Close", "Open"]}},
        {'name': 'deltapy.interact::decision_tree_disc', 'pars': {'cols': ["Close"]}},
        {'name': 'deltapy.interact::quantile_normalize', 'pars': {'drop': ["Close"]}},
        {'name': 'deltapy.interact::tech', 'pars': {}},
        {'name': 'deltapy.interact::genetic_feat', 'pars': {}},

        {'name': 'deltapy.mapper::pca_feature', 'pars': {'variance_or_components': 0.80, 'drop_cols': ["Close_1"]}},
        {'name': 'deltapy.mapper::cross_lag', 'pars': {}},
        {'name': 'deltapy.mapper::a_chi', 'pars': {}},
        {'name': 'deltapy.mapper::encoder_dataset', 'pars': {'drop': ["Close_1"], 'dimesions': 15}},
        {'name': 'deltapy.mapper::lle_feat', 'pars': {'drop': ["Close_1"], 'components': 4}},
        {'name': 'deltapy.mapper::feature_agg', 'pars': {'drop': ["Close_1"], 'components': 4}},
        {'name': 'deltapy.mapper::neigh_feat', 'pars': {'drop': ["Close_1"], 'neighbors': 4}},

        {'name': 'deltapy.extract::abs_energy', 'pars': {}},
        {'name': 'deltapy.extract::cid_ce', 'pars': {'normalize': True}},
        {'name': 'deltapy.extract::mean_abs_change', 'pars': {}},
        {'name': 'deltapy.extract::mean_second_derivative_central', 'pars': {}},
        {'name': 'deltapy.extract::variance_larger_than_standard_deviation', 'pars': {}},
        {'name': 'deltapy.extract::symmetry_looking', 'pars': {}},
        {'name': 'deltapy.extract::has_duplicate_max', 'pars': {}},
        {'name': 'deltapy.extract::partial_autocorrelation', 'pars': {}},
        {'name': 'deltapy.extract::augmented_dickey_fuller', 'pars': {}},
        {'name': 'deltapy.extract::gskew', 'pars': {}},
        {'name': 'deltapy.extract::stetson_mean', 'pars': {}},
        {'name': 'deltapy.extract::length', 'pars': {}},
        {'name': 'deltapy.extract::count_above_mean', 'pars': {}},
        {'name': 'deltapy.extract::longest_strike_below_mean', 'pars': {}},
        {'name': 'deltapy.extract::wozniak', 'pars': {}},
        {'name': 'deltapy.extract::last_location_of_maximum', 'pars': {}},
        {'name': 'deltapy.extract::fft_coefficient', 'pars': {}},
        {'name': 'deltapy.extract::ar_coefficient', 'pars': {}},
        {'name': 'deltapy.extract::index_mass_quantile', 'pars': {}},
        {'name': 'deltapy.extract::number_cwt_peaks', 'pars': {}},
        {'name': 'deltapy.extract::spkt_welch_density', 'pars': {}},
        {'name': 'deltapy.extract::linear_trend_timewise', 'pars': {}},
        {'name': 'deltapy.extract::c3', 'pars': {}},
        {'name': 'deltapy.extract::binned_entropy', 'pars': {}},
        {'name': 'deltapy.extract::svd_entropy', 'pars': {}},
        {'name': 'deltapy.extract::hjorth_complexity', 'pars': {}},
        {'name': 'deltapy.extract::max_langevin_fixed_point', 'pars': {}},
        {'name': 'deltapy.extract::percent_amplitude', 'pars': {}},
        {'name': 'deltapy.extract::cad_prob', 'pars': {}},
        {'name': 'deltapy.extract::zero_crossing_derivative', 'pars': {}},
        {'name': 'deltapy.extract::detrended_fluctuation_analysis', 'pars': {}},
        {'name': 'deltapy.extract::fisher_information', 'pars': {}},
        {'name': 'deltapy.extract::higuchi_fractal_dimension', 'pars': {}},
        {'name': 'deltapy.extract::petrosian_fractal_dimension', 'pars': {}},
        {'name': 'deltapy.extract::hurst_exponent', 'pars': {}},
        {'name': 'deltapy.extract::largest_lyauponov_exponent', 'pars': {}},
        {'name': 'deltapy.extract::whelch_method', 'pars': {}},
        {'name': 'deltapy.extract::find_freq', 'pars': {}},
        {'name': 'deltapy.extract::flux_perc', 'pars': {}},
        {'name': 'deltapy.extract::range_cum_s', 'pars': {}},
        {'name': 'deltapy.extract::structure_func',
         'pars': {'param': {"Volume": df["Volume"].values, "Open": df["Open"].values}}},
        {'name': 'deltapy.extract::kurtosis', 'pars': {}},
        {'name': 'deltapy.extract::stetson_k', 'pars': {}}
    ]

    return prepro_list


def test_deltapy_all2():
    df          = test_get_sampledata()
    prepro_list = test_deltapy_get_method(df)

    for model in prepro_list:
        pars = {'name': model['name'],
                'pars': model['pars']
                }

        df_input = copy.deepcopy(df)
        if 'a_chi' in pars['name']:
            # Normalize the input for the chi, CHi model
            df_input = (df_input - df_input.min()) / (df_input.max() - df_input.min())

        elif 'extract' in pars['name']:
            df_input = df_input["Close"]

        df_out, col_pars = pd_ts_deltapy_generic(df=df_input, pars=pars)



def test_deltapy_all():
    df = test_get_sampledata();
    df.head()

    df_out = transform.robust_scaler(df, drop=["Close_1"])
    df_out = transform.standard_scaler(df, drop=["Close"])
    df_out = transform.fast_fracdiff(df, ["Close", "Open"], 0.5)
    # df_out = transform.windsorization(df,"Close",para,strategy='both')
    df_out = transform.operations(df, ["Close"])
    df_out = transform.triple_exponential_smoothing(df, ["Close"], 12, .2, .2, .2, 0);
    df_out = transform.naive_dec(copy.deepcopy(df), ["Close",
                                                     "Open"])  # The function parameter df is changed within the function causing upcoming functions to crash, passing a copy solves this
    df_out = transform.bkb(df, ["Close"])
    df_out = transform.butter_lowpass_filter(df, ["Close"], 4)
    df_out = transform.instantaneous_phases(df, ["Close"])
    df_out = transform.kalman_feat(df, ["Close"])
    df_out = transform.perd_feat(df, ["Close"])
    df_out = transform.fft_feat(df, ["Close"])
    df_out = transform.harmonicradar_cw(df, ["Close"], 0.3, 0.2)
    df_out = transform.saw(df, ["Close", "Open"])
    df_out = transform.modify(df, ["Close"])
    df_out = transform.multiple_rolling(df, columns=["Close"])
    df_out = transform.multiple_lags(df, start=1, end=3, columns=["Close"])
    df_out = transform.prophet_feat(df.reset_index(), ["Close", "Open"], "Date", "D")

    # **Interaction**
    # The function parameter df is changed within the function causing upcoming functions to crash, passing a copy solves this
    df_out = interact.lowess(copy.deepcopy(df), ["Open", "Volume"], df["Close"], f=0.25, iter=3)
    df_out = interact.autoregression(copy.deepcopy(df))
    df_out = interact.muldiv(copy.deepcopy(df), ["Close", "Open"])
    df_out = interact.decision_tree_disc(copy.deepcopy(df), ["Close"])
    df_out = interact.quantile_normalize(copy.deepcopy(df), drop=["Close"])
    df_out = interact.tech(copy.deepcopy(df))
    df_out = interact.genetic_feat(copy.deepcopy(df))

    # **Mapping**
    df_out = mapper.pca_feature(df, variance_or_components=0.80, drop_cols=["Close_1"])
    df_out = mapper.cross_lag(df)
    '''
    Regarding https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test chi square test assumes frequencies distribution
    and a frequency can't be a negative number. No familiar with the data but if it is safe to either shift them to have min > 0
    or to normalize the data to be [0-1]. Since this is for the purpose of testing we'll be using: (df-df.min())/(df.max()-df.min())
    '''
    df_out = mapper.a_chi((df - df.min()) / (df.max() - df.min()))
    df_out = mapper.encoder_dataset(df, ["Close_1"], 15)
    df_out = mapper.lle_feat(df, ["Close_1"], 4)
    df_out = mapper.feature_agg(df, ["Close_1"], 4)
    df_out = mapper.neigh_feat(df, ["Close_1"], 4)

    # **Extraction**
    extract.abs_energy(df["Close"])
    extract.cid_ce(df["Close"], True)
    extract.mean_abs_change(df["Close"])
    extract.mean_second_derivative_central(df["Close"])
    extract.variance_larger_than_standard_deviation(df["Close"])
    # extract.var_index(df["Close"].values,var_index_param)
    extract.symmetry_looking(df["Close"])
    extract.has_duplicate_max(df["Close"])
    extract.partial_autocorrelation(df["Close"])
    extract.augmented_dickey_fuller(df["Close"])
    extract.gskew(df["Close"])
    extract.stetson_mean(df["Close"])
    extract.length(df["Close"])
    extract.count_above_mean(df["Close"])
    extract.longest_strike_below_mean(df["Close"])
    extract.wozniak(df["Close"])
    extract.last_location_of_maximum(df["Close"])
    extract.fft_coefficient(df["Close"])
    extract.ar_coefficient(df["Close"])
    extract.index_mass_quantile(df["Close"])
    extract.number_cwt_peaks(df["Close"])
    extract.spkt_welch_density(df["Close"])
    extract.linear_trend_timewise(df["Close"])
    extract.c3(df["Close"])
    extract.binned_entropy(df["Close"])
    extract.svd_entropy(df["Close"].values)
    extract.hjorth_complexity(df["Close"])
    extract.max_langevin_fixed_point(df["Close"])
    extract.percent_amplitude(df["Close"])
    extract.cad_prob(df["Close"])
    extract.zero_crossing_derivative(df["Close"])
    extract.detrended_fluctuation_analysis(df["Close"])
    extract.fisher_information(df["Close"])
    extract.higuchi_fractal_dimension(df["Close"])
    extract.petrosian_fractal_dimension(df["Close"])
    extract.hurst_exponent(df["Close"])
    extract.largest_lyauponov_exponent(df["Close"])
    extract.whelch_method(df["Close"])
    extract.find_freq(df["Close"])
    extract.flux_perc(df["Close"])
    extract.range_cum_s(df["Close"])

    '''
    From https://github.com/firmai/deltapy#extraction example, It seems like the second argument of the
    function must be: struct_param = {"Volume":df["Volume"].values, "Open": df["Open"].values}
    '''
    struct_param = {"Volume": df["Volume"].values, "Open": df["Open"].values}
    extract.structure_func(df["Close"], struct_param)
    extract.kurtosis(df["Close"])
    extract.stetson_k(df["Close"])



def test_prepro_v1():
    df         = test_get_sampledata()
    time_eng  = pd_ts_date(df, ['Date'], pars = {})
    onehot    = pd_ts_onehot(df, ['Name'], {})
    trendless = pd_ts_difference(df, ['Close'], {})





########################################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




















"""
The code below extracts all the available features on an example dataset file.

import tsfel
import pandas as pd

# load dataset
df = pd.read_csv('Dataset.txt')

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()

# Extract features
X = tsfel.time_series_features_extractor(cfg, df)
Available features
Statistical domain
Features    Computational Cost
ECDF    1
ECDF Percentile 1
ECDF Percentile Count   1
ECDF Slope  1
Histogram   1
Interquartile range 1
Kurtosis    1
Max 1
Mean    1
Mean absolute deviation 1
Median  1
Median absolute deviation   1
Min 1
Root mean square    1
Skewness    1
Standard deviation  1
Variance    1
Temporal domain
Features    Computational Cost
Absolute energy 1
Area under the curve    1
Autocorrelation 1
Centroid    1
Entropy 1
Mean absolute diff  1
Mean diff   1
Median absolute diff    1
Median diff 1
Negative turning points 1
Peak to peak distance   1
Positive turning points 1
Signal distance 1
Slope   1
Sum absolute diff   1
Total energy    1
Zero crossing rate  1
Neighbourhood peaks 1
Spectral domain
Features    Computational Cost
FFT mean coefficient    1
Fundamental frequency   1
Human range energy  2
LPCC    1
MFCC    1
Max power spectrum  1
Maximum frequency   1
Median frequency    1
Power bandwidth 1
Spectral centroid   2
Spectral decrease   1
Spectral distance   1
Spectral entropy    1
Spectral kurtosis   2
Spectral positive turning points    1
Spectral roll-off   1
Spectral roll-on    1
Spectral skewness   2
Spectral slope  1
Spectral spread 2
Spectral variation  1
Wavelet absolute mean   2
Wavelet energy  2
Wavelet standard deviation  2
Wavelet entropy 2
Wavelet variance    2
Citing


"""







"""

Original file is located at    https://colab.research.google.com/drive/1-uJqGeKZfJegX0TmovhsO90iasyxZYiT

### Introduction
Tabular augmentation is a new experimental space that makes use of novel and traditional data generation and synthesisation techniques to improve model prediction success. It is in essence a process of modular feature engineering and observation engineering while emphasising the order of augmentation to achieve the best predicted outcome from a given information set.
Data augmentation can be defined as any method that could increase the size or improve the quality of a dataset by generating new features or instances without the collection of additional data-points. Data augmentation is of particular importance in image classification tasks where additional data can be created by cropping, padding, or flipping existing images.
Tabular cross-sectional and time-series prediction tasks can also benefit from augmentation. Here we divide tabular augmentation into columnular and row-wise methods. Row-wise methods are further divided into extraction and data synthesisation techniques, whereas columnular methods are divided into transformation, interaction, and mapping methods.
To take full advantage of tabular augmentation for time-series you would perform the techniques in the following order: (1) transforming, (2) interacting, (3) mapping, (4) extracting, and (5) synthesising (forthcoming). What follows is a practical example of how the above methodology can be used. The purpose here is to establish a framework for table augmentation and to point and guide the user to existing packages.
See the [Skeleton Example](#example), for a combination of multiple methods that lead to a halfing of the mean squared error.

Test sets should ideally not be preprocessed with the training data, as in such a way one could be peaking ahead in the training data. The preprocessing parameters should be identified on the test set and then applied on the test set, i.e., the test set should not have an impact on the transformation applied. As an example, you would learn the parameters of PCA decomposition on the training set and then apply the parameters to both the train and the test set.
The benefit of pipelines become clear when one wants to apply multiple augmentation methods. It makes it easy to learn the parameters and then apply them widely. For the most part, this notebook does not concern itself with 'peaking ahead' or pipelines, for some functions, one might have to restructure to code and make use of open source pacakages to create your preferred solution.

pip install deltapy pykalman tsaug ta tsaug pandasvault gplearn ta seasonal pandasvault

Some of these categories are fluid and some techniques could fit into multiple buckets.
This is an attempt to find an exhaustive number of techniques, but not an exhuastive list of implementations of the techniques.

For example, there are thousands of ways to smooth a time-series, but we have only includes 1-2 techniques of interest under each category.



####
date addon
groupby (date, cat2, cat2) statistics.

autoregressive  : past data avg.




(#transformation)**
-----------------
1. Scaling/Normalisation
2. Standardisation
10. Differencing
3. Capping
13. Operations
4. Smoothing
5. Decomposing
6. Filtering
7. Spectral Analysis
8. Waveforms
9. Modifications
11. Rolling
12. Lagging
14. Forecast Model

(#interaction)**
-----------------
1. Regressions
2. Operators
3. Discretising
4. Normalising
5. Distance
6. Speciality
7. Genetic

(#mapping)**
-----------------
1. Eigen Decomposition
2. Cross Decomposition
3. Kernel Approximation
4. Autoencoder
5. Manifold Learning
6. Clustering
7. Neighbouring


(#extraction)**
-----------------
1. Energy
2. Distance
3. Differencing
4. Derivative
5. Volatility
6. Shape
7. Occurence
8. Autocorrelation
9. Stochasticity
10. Averages
11. Size
13. Count
14. Streaks
14. Location
15. Model Coefficients
16. Quantile
17. Peaks
18. Density
20. Linearity
20. Non-linearity
21. Entropy
22. Fixed Points
23. Amplitude
23. Probability
24. Crossings
25. Fluctuation
26. Information
27. Fractals
29. Exponent
30. Spectral Analysis
31. Percentile
32. Range
33. Structural
12. Distribution


"""
