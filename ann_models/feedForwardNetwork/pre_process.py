import numpy as np
import pandas as pd

def create_Xt_Yt(timeS,X, y, percentage=0.70):
    p = int(len(X) * percentage)

    t_train= timeS[0:p]
    X_train = X[0:p]
    Y_train = y[0:p]

    t_test= timeS[p:]
    X_test = X[p:]
    Y_test = y[p:]
    print("len of data ", len(X))
    print("len_trainData", p)
    print("len_testData", len(X) - p)
    print("len time_train: ", len(t_train))
    print("len time_test: ", len(t_test))
    print("train time : ",t_train)
    print("test time : ",t_test)
    print('len X_train : ', len(X_train))
    print('len X_test : ', len(X_test))
    print('len Y_train : ', len(Y_train))
    print('len Y_test : ', len(Y_test))
    print('train data : ', X)
    print("xtrain: ", pd.DataFrame(X_train))
    print("Ytrain: ", pd.DataFrame(Y_train))

    return t_train,t_test, X_train, X_test, Y_train, Y_test

def create_Xt_Yt_Vt(timeS,X, y,v, percentage=0.70):
    p = int(len(X) * percentage)
    print("len_trainData",p)
    print("len_testData", len(X)-p)
    t_train= timeS[0:p]
    X_train = X[0:p]
    Y_train = y[0:p]
    v_train = v[0:p]

    t_test= timeS[p:]
    X_test = X[p:]
    Y_test = y[p:]
    v_test = v[p:]

    return t_train,t_test, X_train, X_test, Y_train, Y_test,v_train,v_test

'''
return newX
'''

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

