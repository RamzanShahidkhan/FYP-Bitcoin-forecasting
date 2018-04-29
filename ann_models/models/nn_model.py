
import numpy as np
from keras.constraints import maxnorm
from keras.models import Input,Sequential,Model
from keras.layers import LSTM
from keras.layers.core import Dense,Activation,Dropout
#from fyp_stockprediction.DeepNeuralNetwork.deepneuralNbitcoin import prepare_data
from keras.optimizers import Adam, SGD
from keras import losses
from ann_models.DeepNeuralNetwork.deepneuralNbitcoin import timestamp_to_dataconv
from ann_models.models.data_preprocess import prepare_data


def dnn_model(minute,closep1,openp,lowp, data, timestamp):
    tr_date, ts_date, x_train, x_test, y_train, y_test = prepare_data(minute,data, timestamp)
    #ts_date = timestamp_to_dataconv(ts_date)

    try:
        # model build in sequential

        main_input = Input(shape=(len(x_train[0]),), name='main_input')
        l = Dense(112, activation='linear', kernel_initializer='normal', kernel_constraint=maxnorm(3))(main_input)
        l = Dense(11, activation='linear', kernel_initializer='normal', kernel_constraint=maxnorm(3))(l)
        l = Dense(8, activation='linear', kernel_initializer='normal', kernel_constraint=maxnorm(3))(l)
        #l = Dense(Dropout(0.5))(l)
        #l = Dense(8, activation='linear')(l)
        #,kernel_initializer='normal', kernel_constraint=maxnorm(3)
        #l = Dense(Dropout(0.2))(l)
        #l = Dense(64, activation='sigmoid')(l)
        output = Dense(1, activation="linear", name="output")(l)
        final_model = Model(inputs=main_input, outputs=output)
        # summary of model
        #print("summaryof model: ", final_model.summary())
        # Compile model
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        opt = Adam(lr=0.01)
        final_model.compile(optimizer=sgd, loss=losses.logcosh)
        # final_model.compile(optimizer=opt,loss='mse' ,metrics=['mse','mae','mape' ])
        # fit the model
        final_model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0
                        )  # validation_data=(X_test, Y_test)
        final_model.evaluate(x_test, y_test, batch_size=256, verbose=0)
        #print("fit end")
        pred = final_model.predict(x_test)

    except:
        print("something is wrong in model")

    predicted = pred
    predicted = predicted.ravel()

    return ts_date,y_test, predicted




def ffnn_model(minute,closep1,openp,lowp, data, timestamp):
    tr_date, ts_date, x_train, x_test, y_train, y_test = prepare_data(minute,data, timestamp)
    #ts_date = timestamp_to_dataconv(ts_date)
    try:
        final_model = Sequential()
        #final_model.load_weights('ann_models/feedForwardNetwork/ffnn-weights',by_name=True)

        # final_model.add(Dropout(0.2, input_shape=(len(x_train[0]),)))
        final_model.add(Dense(112, input_shape=(len(x_train[0]),), kernel_initializer='uniform', activation='linear',
                              kernel_constraint=maxnorm(3)))
        # final_model.add(Dense(112, input_shape=(len(x_train[0]),)))
        final_model.add(Dense(1,kernel_initializer='uniform',kernel_constraint=maxnorm(3), activation="linear"))
        # summary of model
        # print("summaryof model: ", final_model.summary())
        # plot graph
        # plot_model(final_model, to_file="DeepNeuralnetwork.png")
        # Compile model
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
        opt = Adam(lr=0.01)
        #final_model.compile(optimizer=sgd, loss=losses.logcosh)
        final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit the model
        final_model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0,
        validation_data = (x_test, y_test) )
        scores = final_model.evaluate(x_train, y_train,  verbose=0)
        #scores = model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (final_model.metrics_names[1], scores[1] * 100))
        print("fit end")
        # print("weights:  ", final_model.get_weights())
        # print("param:  ",final_model.count_params())
        # print(final_model.__reduce__())
        # print(final_model.legacy_get_config())
        pred = final_model.predict(x_test)

    except:
        print("something is wrong in model")

    predicted = pred
    predicted = predicted.ravel()

    return ts_date,y_test,predicted


def rnn_model(minute,closep1,openp,lowp, data, timestamp):
    tr_date, ts_date, x_train, x_test, y_train, y_test = prepare_data(minute,data, timestamp)
    #ts_date = timestamp_to_dataconv(ts_date)
    print("x_trian[0] shape: ",x_train.shape[0],"-> ", x_train.shape[1])

    x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
    # X_train = X_train.reshape((1,-1))
    # reshape input to be [samples, time steps, features]
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    #x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    print(x_train.shape, x_train.shape, x_test.shape, y_test.shape)

    model = Sequential()
    print("enter in model area")
    # model.add(LSTM(4, input_shape=(X_train)))
    model.add(LSTM(112, input_shape=(1, 5), activation='linear'))
    # model.add(LSTM(4))
    model.add(Dense(1, activation='linear'))
    opt = Adam(lr=0.01)
    model.compile(loss=losses.logcosh, optimizer=opt)
    # print("summaryof model: ", model.summary())
    # plot graph
    # plot_model(model, to_file="DeepNeuralnetwork.png")

    model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0,
              validation_data=(x_test, y_test))  # validation_data=(X_test, Y_test)
    pred = model.predict(x_test)

    predicted = pred
    predicted = predicted.ravel()

    return ts_date, y_test,predicted