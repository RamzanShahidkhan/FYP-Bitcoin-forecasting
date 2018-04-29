from datetime import datetime
from flask import Flask, render_template, request, url_for
import csv
app = Flask(__name__)



def data_read(file_path):
    #file_path = '/home/shahid/PycharmProjects/fyp_web/predicted_data.csv'
    file = open(file_path, newline='')
    myreader = csv.reader(file)
    header = next(myreader)  # first line is the header
    mdates = []
    predicted_prices = []
    actual_prices = []

    for row in myreader:
        #date = datetime.strptime(row[0] ,"%Y-%m-%d %H:%M:%S")#
        date = str(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") ) #
        actual_value = float(row[1])
        predicted_value = float(row[2])
        mdates.append(date)
        actual_prices.append(actual_value)
        predicted_prices.append(predicted_value)

    return mdates, actual_prices, predicted_prices



def errors_read_from_file(file_path):
    file = open(file_path, newline='')
    myreader = csv.reader(file)
    header = next(myreader)  # first line is the header
    for row in myreader:
        rmse = float(row[0])
        mae = float(row[1])
        mape = float(row[2])
    return rmse, mae, mape


ffnn_error_path = 'ann_models/feedForwardNetwork/ffnn_errors_values.csv'
dnn_error_path = 'ann_models/DeepNeuralNetwork/ffnn_errors_values.csv'
rnn_error_path = 'ann_models/RecurrentNetwork/ffnn_errors_values.csv'

ffnn_rmse , ffnn_mae, ffnn_mape = errors_read_from_file(ffnn_error_path)
dnn_rmse , dnn_mae, dnn_mape = errors_read_from_file(dnn_error_path)
rnn_rmse , rnn_mae, rnn_mape = errors_read_from_file(rnn_error_path)
#print(rmse,mae, mape)


ffnn_path = 'ann_models/feedForwardNetwork/predicted_data.csv'
dnn_path = 'ann_models/DeepNeuralNetwork/predicted_data.csv'
rnn_path = 'ann_models/RecurrentNetwork/predicted_data.csv'
ffnn_mydates, ffnn_actual, ffnn_predicted = data_read(ffnn_path)
dnn_mydates, dnn_actual, dnn_predicted = data_read(dnn_path)
rnn_mydates, rnn_actual, rnn_predicted = data_read(rnn_path)

#print(actual)
#print(pred)


@app.route('/')
@app.route('/dashboard')
def hello_world():
    subtitle_label = "ffnn model"
    data_all = zip(ffnn_mydates, ffnn_actual, ffnn_predicted)
    return render_template('ffnn.html', rmse=ffnn_rmse, mae=ffnn_mae,
                           mape=ffnn_mape, data_all=data_all, dates=ffnn_mydates,
                           values=ffnn_actual, values1=ffnn_predicted)


# ffnn
@app.route('/ffnn')
def ffnn():

    subtitle_label = "ffnn model"
    data_all = zip(ffnn_mydates, ffnn_actual, ffnn_predicted)
    return render_template('ffnn.html',rmse=ffnn_rmse , mae=ffnn_mae,
                           mape=ffnn_mape,data_all=data_all,dates=ffnn_mydates,
                           values=ffnn_actual, values1=ffnn_predicted )


# dnn
@app.route('/dnn')
def dnn():

    subtitle_label = "dnn model"
    data_all = zip(dnn_mydates, dnn_actual, dnn_predicted)
    return render_template('dnn.html', rmse=dnn_rmse, mae=dnn_mae,
                           mape=dnn_mape, data_all=data_all, dates=dnn_mydates,
                           values=dnn_actual, values1=dnn_predicted)



# rnn
@app.route('/rnn')
def rnn():
    errors =[rnn_rmse, rnn_mae, rnn_mape]
    subtitle_label = "rnn model"
    data_all = zip(rnn_mydates, rnn_actual, rnn_predicted)
    return render_template('rnn.html',errors=errors, rmse=rnn_rmse, mae=rnn_mae,
                           mape=rnn_mape, data_all=data_all, dates=rnn_mydates,
                           values=rnn_actual, values1=rnn_predicted)



#main method
if __name__ == '__main__':
    app.run(debug=True,port=9922)
