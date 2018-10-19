from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import numpy as np
import pandas as pd


df = pd.read_csv('ANDR1602_clean.csv', sep=',')
df = df.drop(columns=["id", "time step"])

features = ["wind direction", "temperature", "humidity", "pressure",
            "dewpoint", "wind speed at 2 meters", "solar radiation"]
target = ["wind speed"]


series = df[target]


# fit model


#
# # # plot residual errors
# # residuals = DataFrame(model_fit.resid)
# # residuals.plot()
# # pyplot.show()
# # residuals.plot(kind='kde')
# # pyplot.show()
# # print(residuals.describe())
#
X = series.values
size = int(len(X) * 0.9)
# size = 50
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

# SIMPLE ARIMA MODEL
# model = ARIMA(history, order=(5, 1, 0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# predictions, stderr, conf_int = model_fit.forecast(steps=len(test))


## CONTINUOS TRAINING MODEL
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=1)
    yhat = output[0]
    for yhat_i in yhat:
        predictions.append([yhat_i])
    obs = test[t]
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
    # break


mae = mean_absolute_error(test, predictions)
print("Mean Absolute Error: " + str(mae))

mse = mean_squared_error(test, predictions)
print("Mean Squared Error: " + str(mse))

print("Root Mean Squared Error: " + str(math.sqrt(mse)))

r2 = r2_score(test, predictions)
print("R Squared Error: " + str(r2))
#
# # plot
# pyplot.scatter(np.arange(0, len(test), 1), test, color='red')
# pyplot.plot(test, color='blue')
# pyplot.plot(predictions, color='pink')
# pyplot.show()
#
