#########################
# import library/packages
#########################
import numpy as np
import pandas as pd
import os
import mysql.connector
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from calendar import month_abbr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import datetime
import tempfile
import boto3
import joblib
import time

#######################
# load models and files
#######################
# read model from S3 bucket
aws_access_key_id = 'AKIATAVK2UELBEVSLANM'
aws_secret_access_key = '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'

s3_client = boto3.client('s3', 
                         aws_access_key_id=aws_access_key_id, 
                         aws_secret_access_key=aws_secret_access_key)
bucket_name = 'ipowermigrid.monthly.models'

# lr_model load
key = 'linear_monthly.joblib'
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    lr_model = joblib.load(fp)
    
# rf_model load
key = 'random_forest_monthly.joblib'
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    rf_model = joblib.load(fp)
    
# xgb_model load   
key = 'xgboost_model_monthly.joblib'
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    xgb_model = joblib.load(fp)

# xgb_label_encoder load 
key = 'xgb_label_encoder.pkl'
# read model from S3 bucket
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    lbl = joblib.load(fp)

# knn_model load
key = 'knnr_model_monthly.joblib'
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    knn_model = joblib.load(fp)
    
# ensemble_model load    
key = 'ensemble_model_monthly.joblib'
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    ensemble_model = joblib.load(fp)



# load functions
def lagged_data_pred(df, lags):
    df = df#[['end','id', 'demand', 'temp', 'humidity']]
    for i in range(1, lags):
        df["demand_lag_{}".format(i)] = df['demand'].shift(i)
        df["temp_lag_{}".format(i)] = df['temp'].shift(i)
        df["humidity_lag_{}".format(i)] = df['humidity'].shift(i)

    df = pd.DataFrame(df.iloc[-1]).T
    return df

def ModelPredictions(model, X_pred, mg_id):
    prediction = model.predict(X_pred.drop(['end'], axis=1))
    results = pd.DataFrame({'end':X_pred.end,
                            'id':mg_id,
                            'demand':prediction.round(1)  
                           })    
    return results

def XGBModelPredictions(model, X_pred, mg_id, time_index):
    prediction = model.predict(X_pred)
    results = pd.DataFrame({'end':time_index,
                            'id':mg_id,
                            'demand':prediction
                           })    
    return results

###########
# inference
###########

# connect to sql database
credentials = 'mysql://capstone_user:Capstone22!@capstone-database.czwmid1hzf1x.us-west-2.rds.amazonaws.com/mysqldb'

mydb = mysql.connector.connect(
  host="capstone-database.czwmid1hzf1x.us-west-2.rds.amazonaws.com",
  user="capstone_user",
  password="Capstone22!",
  database="mysqldb"
)

mycursor = mydb.cursor()

# set params
mg_id = 'mg_01'

params = {
    'mg_id':mg_id
}


def inference_monthly(mg_id, params=params):
    tail = pd.read_sql('''SELECT * FROM microgrid_actuals_monthly WHERE id = %(mg_id)s ORDER BY end DESC LIMIT 10''', 
                                  con=credentials, params=params)
    # invert the data frame
    tail = tail.iloc[::-1]
    # select the lastest date in the actuals table
    date = tail.iloc[-1]['end']
    # return a fixed frequency DatetimeIndex; grab the lastest date 
    time_index = pd.date_range(date, periods=2, freq='M')[-1]
    # fill in empty record with latest date
    tail.loc[tail.shape[0]] = [time_index, mg_id, np.nan, np.nan, np.nan, np.nan]
    # set prediction month
    tail['month'] = tail['end'].dt.strftime('%b')
    lower_ma = [m.lower() for m in month_abbr]
    tail['month_int'] = tail['month'].str.lower().map(lambda m: lower_ma.index(m)).astype('Int8')

    tail = tail[['end', 'id','demand', 'temp', 'humidity', 'month_int']].copy()

    # lag data
    lr_pred = lagged_data_pred(tail, 3)
    rf_pred = lagged_data_pred(tail, 4)
    xgb_pred = lagged_data_pred(tail, 6)
    knn_pred = lagged_data_pred(tail, 6)

    # prepare model X pred
    lr_X_pred = lr_pred.drop(['id','demand', 'temp', 'humidity'], axis=1)
    rf_X_pred = rf_pred.drop(['id','demand', 'temp', 'humidity'], axis=1)
    xgb_X_pred = xgb_pred.drop(['id','demand', 'temp', 'humidity'], axis=1)
    xgb_X_pred = xgb_X_pred.loc[:, xgb_X_pred.columns != 'end'].astype(float, errors = 'raise')
    xgb_X_pred['month_int'] = lbl.transform(xgb_pred['month_int'].astype(str))
    knn_X_pred = knn_pred.drop(['id','demand', 'temp', 'humidity'], axis=1)

    # prediction results
    lr_results = ModelPredictions(lr_model, lr_X_pred, mg_id).set_index('end').rename(columns={"demand": "lr_demand"}) 
    rf_results = ModelPredictions(rf_model, rf_X_pred, mg_id).set_index('end').rename(columns={"demand": "rf_demand"}) 
    xgb_results = XGBModelPredictions(xgb_model, xgb_X_pred, mg_id, tail['end'].iloc[-1]).set_index('end').rename(columns={"demand": "xgb_demand"}) 
    knn_results = ModelPredictions(knn_model, knn_X_pred, mg_id).set_index('end').rename(columns={"demand": "knn_demand"}) 

    # combine data sets
    frames = [lr_results.lr_demand, rf_results.rf_demand, xgb_results.xgb_demand, knn_results.knn_demand]
    ensemble_X_pred = pd.concat(frames, axis=1, join="inner")
    ensemble_X_pred['month_int'] = tail.iloc[-1].month_int

    # ensemble pred result
    ensemble_X_pred.reset_index(inplace=True)
    ensemble_results = ModelPredictions(ensemble_model, ensemble_X_pred, mg_id)
    ensemble_results.to_sql('microgrid_predictions_monthly', con=credentials, if_exists='append', index=False)

    #time.sleep(5)
    # select the next time step to predict
    actual  = pd.read_sql('''SELECT * FROM microgrid_test_monthly WHERE id = %(mg_id)s ORDER BY end LIMIT 1''', 
                          con=credentials, params=params)
    # write next actual from the test table to the actual table
    actual.to_sql('microgrid_actuals_monthly', con=credentials, if_exists='append', index=False)
    
    # delete updated record from test table
    sql = "DELETE FROM microgrid_test_monthly WHERE id = '%s' AND end = '%s'" % (mg_id, str(actual.iloc[0][0]))
    mycursor.execute(sql)
    mydb.commit()
        
i = 0
while i < 1:
    inference_monthly(mg_id)
    i+=1

