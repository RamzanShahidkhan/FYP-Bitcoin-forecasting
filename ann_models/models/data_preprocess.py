
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pylab as plt
import csv
from keras.initializers import *
import seaborn as sns
plt.style.use('classic')
sns.despine()
from sklearn.preprocessing import MinMaxScaler
from hyperopt import STATUS_OK

def min_max_scalar_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledData = scaler.fit_transform(data)
    np.set_printoptions(precision=3)
    data_original = rescaledData
    return data_original


def prepare_data(minute,data_original, timestamp):
    print("dataaaaaa")
    openp = data_original[:, 1].tolist()[0:]
    highp = data_original[:, 2].tolist()[0:]
    lowp = data_original[:, 3].tolist()[0:]
    closep = data_original[:, 4].tolist()[0:]
    volumep = data_original[:, 5].tolist()[0:]
    #volumecp = data_original[:, 6].tolist()[0:]
    print(openp[0:10])
    x, y = [], []
    x_i = np.column_stack((openp, highp, lowp, closep, volumep))
    print('x_i : ',x_i)
    #print("len x_i", (len(x_i)))
    y_i = closep

    x, y = np.array(x_i), np.array(y_i)
    #print("xxx_array: ",x)
    print('x  : ', len(x), "shape: ", x.shape)
    tr_date, ts_date, X_train, X_test, Y_train, Y_test = create_train_test(minute,timestamp, x, y)
    return tr_date, ts_date, X_train, X_test, Y_train, Y_test



def create_Xt_Yt(timeS,X, y, percentage=0.70):
    p = int(len(X) * percentage)
    print("len_trainData",p)
    print("len_testData", len(X)-p)
    t_train= timeS[0:p]
    X_train = X[0:p]
    Y_train = y[0:p]

    t_test= timeS[p:]
    X_test = X[p:]
    Y_test = y[p:]

    return t_train,t_test, X_train, X_test, Y_train, Y_test



def create_train_test(minute,timeS,X,y,percentage = 0.70):
    p = int(len(X) * percentage)

    t_train = timeS[minute:p]
    X_train = X[0:p-minute]
    Y_train = y[minute:p]

    t_test = timeS[p+minute:]
    X_test = X[p:len(X)-minute]
    Y_test = y[p+minute:]
    print("len of data ", len(X))
    print("len_trainData", p)
    print("len_testData", len(X) - p)
    print("len time_train: ", len(t_train))
    print("len time_test: ", len(t_test))
    print("train time : ", t_train)
    print("test time : ", t_test)
    print('len X_train : ', len(X_train))
    print('len X_test : ', len(X_test))
    print('len Y_train : ', len(Y_train))
    print('len Y_test : ', len(Y_test))
    print('train data : ', X)
    print("xtrain: ", pd.DataFrame(X_train))
    print("Ytrain: ", pd.DataFrame(Y_train))

    return t_train, t_test, X_train, X_test, Y_train, Y_test


