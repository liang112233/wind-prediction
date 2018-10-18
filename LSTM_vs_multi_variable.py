import matplotlib.pyplot as plt
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import numpy as np

np.random.seed(7)



# load dataset
dataset = read_csv('ANDR1602_clean.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 4,5,6,7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group],"k")
	pyplot.title(dataset.columns[group], y=0.5, loc='left')
	i += 1
plt.xlabel("t (5 min)")
pyplot.show()



##################################
###########multi-variable LSTM
################################

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('ANDR1602_clean.csv', header=0, index_col=0)
values = dataset.values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#print(reframed)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
#print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours = 7931
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]



#print(train)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='adam')##previous loss='mae'
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=40, validation_data=(test_X, test_y), verbose=2,shuffle=False)# batch size  72
# plot history
# pyplot.plot(history.history['loss'],"k--", label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction
#yhat = model.predict(test_X)
yhat = model.predict(train_X)

#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
test_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
#print(inv_y)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#plot prediction
fig, ax = plt.subplots()
ax.set_xlim([50, 100])
ax.plot(inv_y,'r')
#ax.plot(test_y,'r')

ax.plot(inv_yhat,'b--')
#ax.plot(testPredictPlot,'g', label='Predict data')
ax.legend(('Real data', 'Prediction'),
           loc='upper right')
plt.xlabel("t (5 min)")
plt.ylabel("v (m/s)")
plt.title('Multi-input LSTM Wind Prediction')
#ax.set_autoscalex_on(False)
#ax.set_xlim([6000,len(values)])
#plt.show()


##########################################################################################
####LSTM
##########################################################################################


import pandas
import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import pandas
#this two lines are for disable the CPU infomation
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# load the dataset

dataframe = pandas.read_csv(r'C:\Users\Fang Jin\PycharmProjects\wind-prediction\ANDR1602_clean.csv', usecols=[1], engine='python', skipfooter=3)
# fix random seed for reproducibility
# plot dataset
dataset = dataframe.values
# plt.plot(dataset)
# plt.show()
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.95)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back),return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=40, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
mae = mean_absolute_error(testY[0], testPredict[:,0])
print('Test MAE: %.3f' % mae)

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
#fig, ax = plt.subplots()
#ax.plot(scaler.inverse_transform(dataset),'r', label='Real data')
#ax.plot(trainPredictPlot,'b', label='Train data')
ax.plot(testPredict,'g--')
ax.legend(('Real data', 'Multi-variable LSTM', 'LSTM'),
           loc='upper right')
plt.xlabel("t (5 min)")
plt.ylabel("v (m/s)")
plt.title('Wind Prediction')
plt.show()







