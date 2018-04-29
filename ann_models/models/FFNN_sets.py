from pandas import Series
import datetime as dt
import numpy as np
from keras.utils import plot_model
import time

np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers import Input
import csv
# plt.rcdefaults()
from keras.optimizers import Adam
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
data_original = pd.read_csv('krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv', usecols=[0, 1, 2, 3, 4, 5])
data_original = data_original.dropna()
# print("description",data_original.describe())
timestamp = data_original.ix[:, 0].tolist()[0:]

closep1 = data_original.ix[:, 4].tolist()[0:]

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledData = scaler.fit_transform(data)
    np.set_printoptions(precision=3)
    return (rescaledData)

data_original = scale_data(data_original)
# print(pd.DataFrame(data_original))
print("shape: ", data_original.shape)

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

def list_powerset(lst):
    # the power set of the empty set has one element, the empty set
    result = [[]]
    for x in lst:
        result.extend([subset + [x] for subset in result])
    return result


def data_sets(labels,p_sets, y_i):
    for i in range(1, len(p_sets), 1):
        ps = p_sets[i]
        label =labels[i]
        x_i = np.column_stack(ps)
        x_i, y_i = np.array(x_i), np.array(y_i)
        #print('ps:  ' + str(i) + "", ps)
        print("len of p: "+str(i)+": ", len(ps))
        print('x_ ' + str(i) + ":  ", x_i)
        tr_date, ts_date, X_train, X_test, Y_train, Y_test = create_Xt_Yt(timestamp, x_i, y_i)

        prepare_model(label,tr_date, ts_date, X_train, X_test, Y_train, Y_test)
        #plt.pause(0.5)
        #time.sleep(0.5)
        #plt.close()


def prepare_data():
    openp = data_original[:, 1].tolist()[0:]
    highp = data_original[:, 2].tolist()[0:]
    lowp = data_original[:, 3].tolist()[0:]
    closep = data_original[:, 4].tolist()[0:]
    volumep = data_original[:, 5].tolist()[0:]

    p_sets = list_powerset([openp, highp, lowp, closep, volumep])
    p_sets_labels = list_powerset(['Open','High','Low','Close','Volume'])
    print("len pset: ", len(p_sets))
    x_i = p_sets
    y_i = closep

    data_sets(p_sets_labels,x_i,y_i)
    print("prepare-data ended !!")


def prepare_model(label,tr_date, ts_date, x_train, x_test, y_train, y_test):
    #tr_date, ts_date, x_train, x_test, y_train, y_test = prepare_data()
    ts_date = timestamp_to_dataconv(ts_date)
    print(x_train.shape, y_train.shape, y_test.shape, y_test.shape)
    print(x_train)

    '''
    print(len(X_train),len(Y_train),len(X_test),len(Y_test))
    '''
    try:
        final_model= Sequential()
        final_model.add(Dense(512, input_shape=(len(x_train[0]),)))
        #final_model.add(Dense(11))
        final_model.add(Dense(1))
        # summary of model
        print("summaryof model: ", final_model.summary())
        # plot graph
        # plot_model(final_model, to_file="DeepNeuralnetwork.png")

        opt = Adam(lr=0.001)
        final_model.compile(optimizer=opt, loss=losses.logcosh)
        # fit the model
        final_model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0
                        )  # validation_data=(X_test, Y_test)
        final_model.evaluate(x_test, y_test, batch_size=256, verbose=0)
        print("fit end")
        #print("weights:  ", final_model.get_weights())
        #print("param:  ",final_model.count_params())
        #print(final_model.__reduce__())
        #print(final_model.legacy_get_config())
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
    #print("prediii: ", len(pred1), "actual: ", len(actual1))
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
    #print("predicted", predicted, "actual", original)
    #print("predicted1: ", pred1, "actual1: ", actual1)

    errors_label = ('mse','rmse', 'mae', 'mape')
    y_pos = np.arange(len(errors_label))
    error_values = np.array([mseA,rmsefA, maefA, mape1A])
    width = 0.75

    tests = ['Layer: 1', 'neurons: {512}', 'activation: {sigmoid,linear}',
             'lr: 0.001']

    plt.figure(1)
    plt.subplot(211)
    # plt.subplot(221)
    plt.xticks(rotation=30)
    plt.plot(ts_date, actual1, label="actual1", color='green')
    plt.plot(ts_date, pred1, label='predicted1', color='red')
    plt.grid(True)
    # plt.plot(Y_train,label ='y_train',color='yellow')
    # plt.plot(Y_test, label='y_test', color='pink')
    plt.title("Bitcoin Price (FeedFNN) inputs "+ str(label), fontweight='bold')
    plt.legend()
    plt.xlabel("Time(s)")
    plt.ylabel("Price of Bitcoin")
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.subplot(223)
    plt.bar(y_pos, error_values, width, align='center', alpha=0.5, color='red',
            )
    plt.xticks(y_pos, errors_label)
    for a, b in zip(y_pos, error_values):
        plt.text(a, b, str(b))
        # plt.annotate(str(b),y_poserror_values=(a,b))
    plt.title('Evaluation Criteria', fontweight='bold')
    plt.xlabel('Errors')
    plt.ylabel('Values')
    plt.subplot(224)
    # plt.subplot(222)
    # plt.subplot(212)
    plt.title("Architecture", fontsize=14)
    # plt.text(.2,.6,'activation function')
    plt.yticks(np.arange(len(tests)) * -1)
    for i, s in enumerate(tests):
        plt.text(0.1, -i / 2, s, fontsize=12, horizontalalignment='left',
                 backgroundcolor='palegreen', wrap=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.show()
    #plt.pause(0.01)
    #time.sleep(0.5)
    plt.close()


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
    prepare_data()
    #prepare_model()
    print(" ended!!!! ")
