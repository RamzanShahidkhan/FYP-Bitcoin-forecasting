from pandas import Series
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
from keras.initializers import *
import seaborn as sns
from ann_models.models.nn_model import ffnn_model,rnn_model,dnn_model

#plt.style.use('classic')
sns.despine()


#timestamp, closep1= timestamp_close_data()


def timestamp_to_dataconv(timestamp):
    dates = np.vectorize(dt.datetime.fromtimestamp)
    dates = dates(timestamp)
    return dates

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledData = scaler.fit_transform(data)
    np.set_printoptions(precision=3)
    return (rescaledData)

def min_max(col):
    minimum = np.min(col)
    maximum = np.max(col)
    return minimum, maximum

#m, ma = min_max(closep1)
#print("min: ", m,"max: ", ma)



def models_result(minute,closep1,openp,lowp, data, timestamp):
    data = scale_data(data)
    print("ffnn")
    ts_date_f, original_f,predicted_f =ffnn_model(minute,closep1,openp,lowp, data, timestamp)
    print("dnn")
    ts_date_d, original_d, predicted_d = dnn_model(minute,closep1,openp,lowp, data, timestamp)
    print("rnn")
    ts_date_r, original_r, predicted_r = rnn_model(minute,closep1,openp,lowp, data, timestamp)

    minimum, maximum = min_max(closep1)
    pred_f = calculate_actual_values(predicted_f, maximum, minimum)
    actual_f = calculate_actual_values(original_f, maximum, minimum)

    pred_d = calculate_actual_values(predicted_d, maximum, minimum)
    actual_d = calculate_actual_values(original_d, maximum, minimum)
    # converted into actual values
    pred_r = calculate_actual_values(predicted_r, maximum, minimum)
    actual_r = calculate_actual_values(original_r, maximum, minimum)

    mseA = np.round(mean_squared_error(actual_f, pred_f),4)
    rmsefA = root_mean_square_error_fun(mseA)
    maefA = mean_absolute_error_fun(actual_f, pred_f)
    mape1A = mean_absolute_percentage_error_fun(actual_f, pred_f)
    r_scoreA = r2_score(actual_f, pred_f)

    print("mse:  ", mseA)
    print("rmse: ", rmsefA)
    print("mae: ", maefA)
    print("mape: ", mape1A)
    print("r2score: ", r_scoreA)

    mse_d = np.round(mean_squared_error(actual_d, pred_d),4)
    rmse_d = root_mean_square_error_fun(mse_d)
    mae_d = mean_absolute_error_fun(actual_d, pred_d)
    mape_d = mean_absolute_percentage_error_fun(actual_d, pred_d)
    r_score_d = r2_score(actual_d, pred_d)

    print("mse:  ", mse_d,"rmse: ", rmse_d,"mae: ", mae_d,"mape: ", mape_d,
          "r2score: ", r_score_d)

    mse_r = np.round(mean_squared_error(actual_r, pred_r),2)
    rmse_r = root_mean_square_error_fun(mse_r)
    mae_r = mean_absolute_error_fun(actual_r, pred_r)
    mape_r = mean_absolute_percentage_error_fun(actual_r, pred_r)
    r_score_r = r2_score(actual_r, pred_r)

    print("mse:  ", mse_r, "rmse: ", rmse_r, "mae: ", mae_r, "mape: ", mape_r,
          "r2score: ", r_score_r)

    errors_label = ('rmse', 'mae', 'mape')
    y_pos = np.arange(len(errors_label))
    error_values = [rmsefA, maefA, mape1A]
    #error_values_d = np.array([mse_d, rmse_d, mae_d, mape_d])
    error_values_d = [rmse_d, mae_d, mape_d]
    error_values_r = [rmse_r, mae_r, mape_r]
    width = 0.25

    tests = ['Hidden Layers: 2', 'neurons: {512,11}', 'activation: {sigmoid,linear}',
             'lr: 0.001']
    ind = np.arange(3)
    plt.figure(1)
    plt.subplot(211)
    plt.xticks(rotation=30)
    plt.plot(ts_date_f, actual_f, label='close', color='red')
    plt.plot(ts_date_f, pred_f, label="ffnn", color='green')
    plt.plot(ts_date_f, pred_d, label="rnn", color='yellow')
    plt.plot(ts_date_f, pred_r, label="dnn", color='blue')

    plt.grid(True)
    plt.title("Forecast price "+str(minute) +" minutes", fontweight='bold')
    plt.legend(loc=2) # to left position
    plt.xlabel("Time")
    plt.ylabel("Price of Bitcoin (USD)")
    plt.subplot(212)
    rect1 = plt.bar(ind, error_values, width,align='center', color='r',label='ffnn')
    rect2 = plt.bar(ind+width+0.00, error_values_d,  width,align='center',color='b',label='dnn')
    rect3 = plt.bar(ind+(width*2)+0.01, error_values_r,width, align='center',color='g',label='rnn')
    plt.xticks(y_pos, errors_label)
    plt.grid(axis='y')
    plt.legend()
    #plt.legend((error_values[0],error_values_d[0],error_values_r[0]),('FFNN','DNN','RNN'))

    for a, f,d,r in zip(y_pos, error_values,error_values_d,error_values_r):
        plt.text(a, f, str(f))
        plt.text(a + 0.25, d, str(d))
        plt.text(a + .50, r, str(r))
        # plt.annotate(str(b),y_poserror_values=(a,b))

    plt.title('Evaluation Criteria', fontweight='bold')
    plt.xlabel('Errors')
    plt.ylabel('Values of Errors')
    '''
    '''
    plt.subplots_adjust(hspace=0.7, wspace=0.7)

    plt.show()

    return 0


# invert values into to actual values
def calculate_actual_values(y, max, min):
    dumer = max - min
    x = (y * dumer) + min
    return x


# RMSE
def root_mean_square_error_fun(value):
    rmse = np.sqrt(value)
    return np.round(rmse, 2)


# MAE
def mean_absolute_error_fun(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    return np.round(mae, 2)


# MAPE
def mean_absolute_percentage_error_fun(y_true, y_pred):
    actual = np.array(y_true)
    predicted = np.array(y_pred)
    sum1 = 0
    count = len(actual)
    for i in range(count):
        forecast_error = abs((actual[i] - predicted[i])) / actual[i]
        sum1 = sum1 + forecast_error
    mape = (sum1 / count) * 100
    return np.round(mape, 2)

# writing actual and predicted values

def write_values(actual,predicted):

    file = open('actualfile.csv','w')
    with file:
        writter = csv.writer(file)
        writter.writerow(predicted.ravel())
    with open('actualfile.csv', 'a') as f:
        #f.write(str(pred1.ravel()))
        f.write(str(actual.ravel()))



def data_load_fun():
    # file_path = 'ann_models/data_files/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv'
    # file_kraken = '/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv'
    # file_coincheck = '/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/coincheckJPY_1-min_data_2014-10-31_to_2018-03-27.csv'
    # file_bitstamp = '/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv'
    file_btceUSD = '/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv'
    data_original = pd.read_csv(file_btceUSD)
    data = data_original.dropna()
    openp = data['Open']
    lowp = data['Low']
    minute = 20

    timestamp = pd.to_datetime(data.Timestamp, unit='s')
    closep1 = data.ix[:, 4].tolist()[0:]
    # print(data)
    models_result(minute, closep1, openp, lowp, data, timestamp)
    ## print(" ended!!!! ")
    #return t, p, a


data_load_fun()