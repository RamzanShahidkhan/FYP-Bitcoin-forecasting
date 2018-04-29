from keras.constraints import maxnorm
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
from keras.layers import Input
import csv
# plt.rcdefaults()
from keras.optimizers import Adam, SGD
from keras.initializers import *
from keras import losses
from keras import backend as K
import seaborn as sns
from ann_models.feedForwardNetwork.pre_process import create_Xt_Yt

#plt.style.use('classic')
sns.despine()
from sklearn.preprocessing import MinMaxScaler
from hyperopt import STATUS_OK


# data_original = pd.read_csv('krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv',usecols=['Open','High','Low','Close','Volume_(BTC)','Volume_(Currency)'])[-1000:]
# data_original = pd.read_csv('btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv',usecols=[1,2,3,4,5])


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


def prepare_data(data_original, timestamp):
    openp = data_original[:, 1].tolist()[0:]
    highp = data_original[:, 2].tolist()[0:]
    lowp = data_original[:, 3].tolist()[0:]
    closep = data_original[:, 4].tolist()[0:]
    volumep = data_original[:, 5].tolist()[0:]

    x, y = [], []

    x_i = np.column_stack((openp, highp, lowp, closep, volumep))
    print("len", (len(x_i)))
    # print("X_i",x_i)
    y_i = closep
    # print("closeP: ",pd.DataFrame(y_i))
    x, y = np.array(x_i), np.array(y_i)
    print(len(x), len(y))
    # print("X",X)
    tr_date, ts_date, X_train, X_test, Y_train, Y_test = create_Xt_Yt(timestamp, x, y)
    return tr_date, ts_date, X_train, X_test, Y_train, Y_test
    print("fun prepare_Data ended!!! ")


def prepare_model(closep1,openp,lowp, data, timestamp):
    data = scale_data(data)
    p = int(len(lowp) * 0.70)
    openp = openp[p:]
    lowp = lowp[p:]
    tr_date, ts_date, x_train, x_test, y_train, y_test = prepare_data(data,timestamp)
    #ts_date = timestamp_to_dataconv(ts_date)
    print(x_train.shape, y_train.shape, y_test.shape, y_test.shape)
    #print(x_train)
    try:
        final_model = Sequential()
        #final_model.add(Dropout(0.2, input_shape=(len(x_train[0]),)))
        final_model.add(Dense(112,input_shape=(len(x_train[0]),), kernel_initializer='normal', activation='linear', kernel_constraint=maxnorm(3)))
        #final_model.add(Dense(112, input_shape=(len(x_train[0]),)))
        # final_model.add(Dense(11))
        final_model.add(Dense(1, activation="linear"))
        # summary of model
        #print("summaryof model: ", final_model.summary())
        # plot graph
        # plot_model(final_model, to_file="DeepNeuralnetwork.png")
        # Compile model
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        opt = Adam(lr=0.01)
        final_model.compile(optimizer=sgd, loss=losses.logcosh)
        # fit the model
        final_model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0
                        )  # validation_data=(X_test, Y_test)
        final_model.evaluate(x_test, y_test, batch_size=256, verbose=0)
        print("fit end")

        '''
        # save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

        # some time later...

        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, Y_test)
        '''
        final_model.save_weights("ffnn-weights")

        # print("weights:  ", final_model.get_weights())
        # print("param:  ",final_model.count_params())
        # print(final_model.__reduce__())
        # print(final_model.legacy_get_config())
        pred = final_model.predict(x_test)

    except:
        print("something is wrong in model")

    predicted = pred
    predicted = predicted.ravel()
    original = y_test
    # actual converted values after model
    minimum, maximum = min_max(closep1)
    pred1 = calculate_actual_values(predicted, maximum, minimum)
    actual1 = calculate_actual_values(original, maximum, minimum)


    mseA = mean_squared_error(actual1, pred1)
    rmsefA = root_mean_square_error_fun(mseA)
    maefA = mean_absolute_error_fun(actual1, pred1)
    mape1A = mean_absolute_percentage_error_fun(actual1, pred1)
    r_scoreA = r2_score(actual1, pred1)

    # writing data in csv
    write_predicted_data(ts_date, actual1, pred1)
    write_errors_in_file(rmsefA,maefA,mape1A)

    print("mse:  ", mseA)
    print("rmse: ", rmsefA)
    print("mae: ", maefA)
    print("mape: ", mape1A)
    print("r2score: ", r_scoreA)

    errors_label = ('rmse', 'mae', 'mape')
    y_pos = np.arange(len(errors_label))
    error_values = np.array([rmsefA, maefA, mape1A])
    width = 0.75

    plt.figure(1)
    # plt.subplot(211)
    plt.subplot(221)
    plt.xticks(rotation=30)

    plt.plot(ts_date, actual1, label="Close Price", color='green')
    #plt.plot(ts_date, lowp, label='Low', color='blue')
    #plt.plot(ts_date, openp, label='Open', color='yellow')
    plt.plot(ts_date, pred1, label='Predicted', color='red')

    plt.grid(True)
    plt.title("Bitcoin Price (FFNN)", fontweight='bold')
    plt.legend(loc=2)
    plt.xlabel("Time")
    plt.ylabel("Price of Bitcoin (USD)")
    # plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.subplot(222)
    # plt.subplot(223)
    plt.bar(y_pos, error_values, width, align='center', alpha=0.5, color='red',
            )

    plt.xticks(y_pos, errors_label)
    for a, b in zip(y_pos, error_values):
        plt.text(a, b, str(b))
        # plt.annotate(str(b),y_poserror_values=(a,b))

    plt.title('Evaluation Criteria', fontweight='bold')
    plt.xlabel('Errors')
    plt.ylabel('Values')
    plt.subplot(212)
    # plt.subplot(222)
    # plt.subplot(212)
    plt.title("Architecture of Model", fontsize=14)
    # plt.yticks(np.arange(len(tests)) * -1)
    # for i, s in enumerate(tests):
    #   plt.text(0.1,-i/2, s,fontsize=12,horizontalalignment='left',
    #           backgroundcolor='palegreen',wrap=True)
    plt.text(0.025, 0.8, "Layers :- ", fontweight='bold')
    plt.text(0.2, 0.8, "Two layers", fontsize=10)

    plt.text(0.025, 0.6, "Activation Fun :- ", fontweight='bold')
    plt.text(0.2, 0.6, "{ sigmoid, linear }")

    plt.text(0.025, 0.4, "Data_Set :- ", fontweight='bold')
    plt.text(0.2, 0.4, "Training: 70% and Testing: 30%, inputs: 5 ")

    plt.text(0.025, 0.2, "Features :- ", fontweight='bold')
    plt.text(0.2, 0.2, "batchsize: 256,  epochs: 5,  lr: 0.01, Optimizer: Adam")

    plt.subplots_adjust(hspace=0.6, wspace=0.5, left=0.125, bottom=0.1, right=None, top=None)

    # Save Figure
    #plt.savefig("foo.png")
    # Save Transparent Figure
    #plt.savefig("foo.png", transparent=True)

    plt.show()

    print("ended model process")
    return ts_date, pred1, actual1


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


def write_predicted_data(timestamp ,actual,predicted):
    # open a file for writing.
    file = open('predicted_data.csv', 'w')
    # create the csv writer object.
    mywriter = csv.writer(file)
    mywriter.writerow(['Date','Actual', 'Predicted'])
    # all rows at once.
    for row in zip(timestamp,actual,predicted): # write data in cols wise
        mywriter.writerow(row)
    # writing in rows wise
    #mywriter.writerow(actual)
    #mywriter.writerow(predicted)
    #prices = [actual, predicted]
    #mywriter.writerows(prices) # write data in row wise
    file.close()

def write_errors_in_file( rmse_value,mae_value,mape_value):
    # open a file for writing.
    file = open('ffnn_errors_values.csv', 'w')
    # create the csv writer object.
    mywriter = csv.writer(file)
    mywriter.writerow(['RMSE','MAE', 'MAPE'])
    mywriter.writerow([rmse_value,mae_value,mape_value])

    file.close()


def data_load_fun():
    file_path = 'ann_models/data_files/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv'
    file_kraken ='/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv'
    file_coincheck ='/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/coincheckJPY_1-min_data_2014-10-31_to_2018-03-27.csv'
    file_bitstamp = '/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv'
    file_btceUSD = '/home/shahid/PycharmProjects/fyp_bitcoin_prediction_ANNs/ann_models/data_files/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv'

    data_original = pd.read_csv(file_btceUSD)
    data = data_original.dropna()
    openp =data['Open']
    lowp = data['Low']

    timestamp = pd.to_datetime(data.Timestamp, unit='s')
    closep1 = data.ix[:, 4].tolist()[0:]

    t,p, a = prepare_model(closep1,openp,lowp, data, timestamp)
    ## print(" ended!!!! ")
    return t, p, a

data_load_fun()
