import numpy
import pandas as pd
import matplotlib.pyplot as plt

# load the data set
data_original = pd.read_csv('./krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv', usecols=[1,2,3,4,5,6],engine='python')
data_original = data_original.dropna()
data_original = data_original.values
#data_original = data_original.astype('float32')
print("ddd ",data_original[3])
print(data_original.shape)
train_size = int(len(data_original)* 0.70)
test_size = len(data_original) - train_size
print(train_size, test_size)
train = data_original[0:train_size,:]
test =  data_original[train_size:len(data_original),:]
#test =  data_original[train_size:len(data_original),4]

#,percentage =0.67
def create_Xt_Yt(X,y):
    p = (int(len(X) )*.70)
    X_train = X[0:p]
    Y_train = y[0:p]
    #shuffle
    X_test = X[p:]
    Y_test = y[p:]
    return X_train, X_test, Y_train, Y_test

#trx,tsx,try1, tsy1 = create_Xt_Yt(data_original,data_original[3])
#print("jajja", len(trx), len(tsx))
dataX, dataY = [], []


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)



look_back = 1
trainX, trainY= create_dataset(train, look_back)
testX, testY = create_dataset(test,look_back)
train = pd.DataFrame(train)
test = pd.DataFrame(test)
trainX = pd.DataFrame(trainX)
trainY = pd.DataFrame(trainY)
print("len data : ",len(data_original))
print("datax ",len(dataX))
print("dataY : ",len(dataY))
print("l train: ",len(train))
print("l test: ",len(test))
print("l xtrain: ",len(trainX))
print("l testX: ",len(testX))
print("l yrain: ",len(trainY))
print("l testY: ",len(testY))

print("tarin: ", train)
print("test : ", test)
print("trainX")
print(trainX)
print("trainY")
print(trainY)
'''
plt.plot(data_original, label="original")
plt.plot(train, label="train")
plt.plot(test, label ="test")
plt.legend()
plt.show()

'''