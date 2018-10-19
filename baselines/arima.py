from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('ANDR1602_clean.csv', sep=',')
# df = df.drop(columns=["id", "time step"])

features = ["wind direction", "temperature", "humidity", "pressure",
            "dewpoint", "wind speed at 2 meters", "solar radiation"]
target = ["wind speed"]


samples = 8352
test_size = int(0.1*samples)
idx_list = np.linspace(0, samples-1, num=samples)
idx_train = np.arange(0, samples-test_size)
idx_test = np.delete(idx_list, idx_train).astype('int')

series = df[target]
X = series.values
train, test = X[idx_train], X[idx_test]
history = [x for x in train]

# SIMPLE ARIMA MODEL
model = ARIMA(history, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
testPredict, stderr, conf_int = model_fit.forecast(steps=len(test))


## CONTINUOS TRAINING MODEL
# predictions = list()
# for t in range(len(test)):
#     model = ARIMA(history, order=(5,1,0))
#     model_fit = model.fit(disp=0)
#     output, stderr, conf_int = model_fit.forecast(steps=1)
#     yhat = output
#     predictions.append([yhat])
#     obs = test[t]
#     history.append(obs)
#     # print('predicted=%f, expected=%f' % (yhat, obs))
#     # break


mae = mean_absolute_error(test, testPredict)
print("Mean Absolute Error: " + str(mae))

mse = mean_squared_error(test, testPredict)
print("Mean Squared Error: " + str(mse))

print("Root Mean Squared Error: " + str(math.sqrt(mse)))

r2 = r2_score(test, testPredict)
print("R Squared Error: " + str(r2))
#
# # plot
# pyplot.scatter(np.arange(0, len(test), 1), test, color='red')
# pyplot.plot(test, color='blue')
# pyplot.plot(predictions, color='pink')
# pyplot.show()
#

x_indices = [i for i in range(len(idx_test))]

plt.plot(x_indices, test, color='blue')
plt.plot(x_indices, testPredict, color='pink')
plt.show()