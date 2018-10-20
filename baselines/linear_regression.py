from sklearn import linear_model
import pandas as pd
from sklearn import datasets ## imports datasets from scikit-learn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

# np.random.seed(7)

samples = 8352
train_size = int(0.9*samples)
# # create training and testing sets
# idx_list = np.linspace(0, samples-1, num=samples)
# idx_test = np.random.choice(samples, size=test_size, replace=False)
# idx_train = np.delete(idx_list, idx_test).astype('int')

idx_train = [i for i in range(train_size)]
idx_test = [j for j in range(train_size, samples)]

df = pd.read_csv('ANDR1602_clean.csv', sep=',')
# df = df.drop(columns=["id", "time step"])

features = ["wind direction", "temperature", "humidity", "pressure",
            "dewpoint", "wind speed at 2 meters", "solar radiation"]
target = ["wind speed"]

df_train = df.iloc[idx_train, ]
df_test = df.iloc[idx_test, ]

X_train = df_train[features]
y_train = df_train[target]


lm = linear_model.LinearRegression(normalize=True)
model = lm.fit(X_train, y_train)

X_test = df_test[features]
y_test_true = df_test[target]

testPredict = lm.predict(X_test)
print(testPredict[0:5])


mse = mean_squared_error(y_test_true, testPredict)
print("Mean Squared Error: " + str(mse))

print("Root Mean Squared Error: " + str(math.sqrt(mse)))

mae = mean_absolute_error(y_test_true, testPredict)
print("Mean Absolute Error: " + str(mae))

r2 = r2_score(y_test_true, testPredict)
print("R Squared Error: " + str(r2))

# plt.scatter(X_test, y_test_true, color='red')
x_indices = [i for i in range(len(idx_test))]

# plt.plot(x_indices, y_test_true, color='blue')
# plt.plot(x_indices, testPredict, color='pink')
# plt.show()

plot_results(testPredict, y_test_true.values)