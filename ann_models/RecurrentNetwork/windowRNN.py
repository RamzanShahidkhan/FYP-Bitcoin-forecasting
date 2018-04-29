from pandas import Series
import datetime as dt
import matplotlib.dates as mdates
from sklearn.utils import check_array
import numpy as np
from keras.utils import plot_model

np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers import Input, LSTM
import csv
# plt.rcdefaults()
from keras.optimizers import Adam
from keras.initializers import *
from keras import losses
from keras import backend as K
import seaborn as sns
from fyp_stockprediction.feedForwardNetwork.utils import create_Xt_Yt

plt.style.use('classic')
sns.despine()
from sklearn.preprocessing import MinMaxScaler
from hyperopt import STATUS_OK

# data_original = pd.read_csv('krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv',usecols=['Open','High','Low','Close','Volume_(BTC)','Volume_(Currency)'])[-1000:]
# data_original = pd.read_csv('btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv',usecols=[1,2,3,4,5])
data_original = pd.read_csv('krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv', usecols=[0, 1, 2, 3, 4, 5,6])
data_original = data_original.dropna()
# print("description",data_original.describe())
timestamp = data_original.ix[:, 0].tolist()[0:]

closep1 = data_original.ix[:, 4].tolist()[0:]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledData = scaler.fit_transform(data_original)
np.set_printoptions(precision=3)
data_original = rescaledData
# print(pd.DataFrame(data_original))
print("shape: ", data_original.shape)

STEP = 1
FORECAST = 10

def timestamp_to_dataconv(timestamp):
    dates = np.vectorize(dt.datetime.fromtimestamp)
    dates = dates(timestamp)
    return dates


def min_max(col):
    minimum = np.min(col)
    maximum = np.max(col)
    return minimum, maximum

m, ma = min_max(closep1)
print("min: ", m,"max: ", ma)


def prepare_data(window):
    openp = data_original[:, 1].tolist()[0:]
    highp = data_original[:, 2].tolist()[0:]
    lowp = data_original[:, 3].tolist()[0:]
    closep = data_original[:, 4].tolist()[0:]
    volumep = data_original[:, 5].tolist()[0:]
    volumecp = data_original[:, 6].tolist()[0:]
    volatility = pd.DataFrame(closep).rolling(window).std().values.tolist()
    volatility = [v[0] for v in volatility]
    x, y = [], []

    for i in range(0, len(data_original), STEP):
        try:
            o = openp[i:i + window]
            h = highp[i:i + window]
            l = lowp[i:i + window]
            c = closep[i:i + window]
            v = volumep[i:i + window]
            vc = volumecp[i:i + window]
            volat = volatility[i:i + window]

            o = (np.array(o) - np.mean(o)) / np.std(o)
            h = (np.array(h) - np.mean(h)) / np.std(h)
            l = (np.array(l) - np.mean(l)) / np.std(l)
            c = (np.array(c) - np.mean(c)) / np.std(c)
            v = (np.array(v) - np.mean(v)) / np.std(v)

            vc = (np.array(vc) - np.mean(vc)) / np.std(vc)
            volat = (np.array(volat) - np.mean(volat)) / np.std(volat)

            x_i = np.column_stack((o, h, l, c, v, vc, volat))
            #x_i = x_i.flatten()
            y_i = (closep[i + window + FORECAST] - closep[i + window]) / closep[i + window]

            if np.isnan(x_i).any():
                continue

        except Exception as e:
            break

        x.append(x_i)
        y.append(y_i)

    x, y = np.array(x), np.array(y)
    # print("X",X)
    X_train, X_test, Y_train, Y_test = create_Xt_Yt(x, y)
    return X_train, X_test, Y_train, Y_test


def prepare_model():
    x_train, x_test, y_train, y_test = prepare_data(30)
    #ts_date = timestamp_to_dataconv(ts_date)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print("xlentrain: ", len(x_train))
    # x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    # y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
    # x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
    #print("xtrain: ", x_train)
    #print("ytrain: ", y_train)
    #print("xtest: ", x_test)
    # X_train = X_train.reshape((1,-1))

    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # Y_train = Y_train.reshape((Y_train.shape[0],1,Y_train.shape[1]))

    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    print(x_train.shape, x_train.shape, x_test.shape, y_test.shape)

    model = Sequential()
    print("enter in model area")
    # model.add(LSTM(4, input_shape=(X_train)))
    model.add(LSTM(4, input_shape=(1, 6)))
    # model.add(LSTM(4))
    model.add(Dense(1, activation='sigmoid'))
    print("pass layer")
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    print("pass compile layer")
    print("summaryof model: ", model.summary())
    # plot graph
    # plot_model(model, to_file="DeepNeuralnetwork.png")
    print("entering in fit....")
    model.fit(x_train, y_train, epochs=5, batch_size=200, verbose=2,
              validation_data=(x_test, y_test))  # validation_data=(X_test, Y_test)
    print("pass the fit phase")
    pred = model.predict(x_test)


    predicted = pred
    predicted = predicted.ravel()
    original = y_test
    # actual converted values after model
    minimum, maximum = min_max(closep1)
    pred1 = calculate_actual_values(predicted, maximum, minimum)
    actual1 = calculate_actual_values(original, maximum, minimum)

    print("prediii: ", len(pred1), "actual: ", len(actual1))
    '''
    file = open('actualfile.csv','w')
    with file:
        writter = csv.writer(file)
        writter.writerow(pred1.ravel())
    with open('actualfile.csv', 'a') as f:
        #f.write(str(pred1.ravel()))
        f.write(str(actual1.ravel()))
        '''
    # actual values
    mseA = mean_squared_error(actual1, pred1)
    rmsefA = root_mean_square_error_fun(mseA)
    maefA = mean_absolute_error_fun(actual1, pred1)
    mape1A = mean_absolute_percentage_error_fun(actual1, pred1)
    r_scoreA = r2_score(actual1, pred1)

    print("mse:  ", mseA)
    print("rmse: ", rmsefA)
    print("mae: ", maefA)
    print("mape: ", mape1A)
    print("r2score: ", r_scoreA)
    print("predicted", predicted, "actual", original)
    print("predicted1: ", pred1, "actual1: ", actual1)
    Y_train = calculate_actual_values(y_train, maximum, minimum)
    Y_test = calculate_actual_values(y_test, maximum, minimum)

    errors_label = ('mse','rmse', 'mae', 'mape')
    y_pos = np.arange(len(errors_label))
    error_values = np.array([mseA,rmsefA, maefA, mape1A])
    width = 0.75

    plt.figure(1)
    plt.subplot(211)
    # plt.subplot(221)
    plt.bar(y_pos, error_values, width, align='center', alpha=0.5, color='red')
    plt.xticks(y_pos, errors_label)
    for a, b in zip(y_pos, error_values):
        plt.text(a, b, str(b))
        # plt.annotate(str(b),y_poserror_values=(a,b))
    plt.title('Evaluation Criteria')
    plt.xlabel('Errors')
    plt.ylabel('Values')
    plt.subplot(223)
    plt.subplot(224)
    # plt.subplot(222)
    # plt.subplot(212)
    plt.xticks(rotation=30)
    plt.plot(actual1, label="actual1", color='green')
    plt.plot( pred1, label='predicted1', color='red')
    # plt.plot(Y_train,label ='y_train',color='yellow')
    # plt.plot(Y_test, label='y_test', color='pink')
    plt.title("Prediction of Bitcoin Price (Recurrent_NN")
    plt.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Price of Bitcoin")
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    # plt.text(4,"kalia")

    mse = mean_squared_error(original, predicted)
    print("msee: ", mse)
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
    return np.round(rmse, 4)


# MAE
def mean_absolute_error_fun(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    return np.round(mae, 4)


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
    return np.round(mape, 4)


if __name__ == '__main__':
    #data_original = pd.read_csv('krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv', usecols=[0, 1, 2, 3, 4, 5])
    #data_original = data_original.dropna()
    #prepare_data(data_original)
    prepare_model()
    print(" ended!!!! ")
