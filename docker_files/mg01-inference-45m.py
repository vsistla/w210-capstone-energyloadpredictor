import numpy as np
import pandas as pd
import os
import mysql.connector
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from calendar import month_abbr
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import mean_squared_error
import datetime
import tempfile
import boto3
import joblib


aws_access_key_id = 'AKIATAVK2UELBEVSLANM'
aws_secret_access_key = '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'

s3_client = boto3.client('s3', 
                         aws_access_key_id=aws_access_key_id, 
                         aws_secret_access_key=aws_secret_access_key)
bucket_name = 'mg01-models'
key = 'linear_model_60m.pkl'

# read from 
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    lr_from_s3 = joblib.load(fp)



def lagged_data_pred(df):
    df = df#[['end','id', 'demand', 'temp', 'humidity']]
    for i in range(1, 10):
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
####################################
mg_id = 'mg_01'

params = {
    'mg_id':mg_id
}
####################################

def inference(mg_id, number_of_predictions=1, params=params):
    for i in range(number_of_predictions):
        tail = pd.read_sql('''SELECT * FROM microgrid_actuals_45m WHERE id = %(mg_id)s ORDER BY end DESC LIMIT 10''', 
                                      con=credentials, params=params)
        # invert the data frame
        tail = tail.iloc[::-1]
        # select the lastest date in the actuals table
        date = tail.iloc[-1]['end']
        # return a fixed frequency DatetimeIndex; grab the lastest date 
        time_index = pd.date_range(date, periods=46, freq='min')[-1]
        # fill in empty record with latest date
        tail.loc[tail.shape[0]] = [time_index, mg_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        tail['month'] = tail['end'].dt.strftime('%b')

        lower_ma = [m.lower() for m in month_abbr]

        # one-liner with Pandas
        tail['month_int'] = tail['month'].str.lower().map(lambda m: lower_ma.index(m)).astype('Int8')

        tail['day_of_week'] = tail['end'].dt.day_name()

        tail['day_of_week_int'] = tail['end'].dt.day_of_week

        date_range = pd.date_range(start='2019-01-01', end='2022-01-27')

        cal = calendar()
        holidays = cal.holidays(start=date_range.min(), end=date_range.max())

        tail['holiday'] = tail['end'].dt.date.astype('datetime64').isin(holidays)

        tail["holiday_int"] = tail["holiday"].astype(int)

        tail = tail[['end', 'id','demand', 'temp', 'humidity', 'month_int', 'day_of_week_int', 'holiday_int']].copy() 

        # transform records to lagged data format
        pred = lagged_data_pred(tail)

        X_pred = pred.drop(['id','demand', 'temp', 'humidity'], axis=1)

        # set predict value
        results = ModelPredictions(lr_from_s3, X_pred, mg_id)
        # write results to sql table
        results.to_sql('microgrid_predictions_45m', con=credentials, if_exists='append', index=False)

        # select the next time step to predict
        actual  = pd.read_sql('''SELECT * FROM microgrid_test_45m WHERE id = %(mg_id)s ORDER BY end LIMIT 1''', 
                              con=credentials, params=params)
        # write next actual from the test table to the actual table
        actual.to_sql('microgrid_actuals_45m', con=credentials, if_exists='append', index=False)
        # delete updated record from test table
        sql = "DELETE FROM microgrid_test_45m WHERE id = '%s' AND end = '%s'" % (mg_id, str(actual.iloc[0][0]))
        mycursor.execute(sql)
        mydb.commit()
        
inference(mg_id, 1)