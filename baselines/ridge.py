
from sklearn import linear_model
import pandas as pd
from sklearn import datasets ## imports datasets from scikit-learn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import math
from baselines.util import plot_results

np.random.seed(7)

samples = 8352
train_size = int(0.9*samples)
idx_train = [i for i in range(train_size)]
idx_test = [j for j in range(train_size, samples)]

# test_size = int(0.1*samples)
# create training and testing sets
# idx_list = np.linspace(0, samples-1, num=samples)
# idx_test = np.random.choice(samples, size=test_size, replace=False)
# idx_train = np.delete(idx_list, idx_test).astype('int')

df = pd.read_csv('ANDR1602_clean.csv', sep=',')
# df = df.drop(columns=["id", "time step"])

features = ["wind direction", "temperature", "humidity", "pressure",
            "dewpoint", "wind speed at 2 meters", "solar radiation"]
target = ["wind speed"]

df_train = df.iloc[idx_train, ]
df_test = df.iloc[idx_test, ]

X_train = df_train[features]
y_train = df_train[target]

#### Core model #####
lm = linear_model.Ridge(normalize=True, max_iter=1e5)
model = lm.fit(X_train, y_train)

regr_score = lm.score(X_train, y_train)
# regr_alpha = lm.alpha_
print('score:', regr_score)
print('CV', lm.coef_)

#### ########################

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
# plt.plot(idx_test, y_test_true, color='blue')
# plt.plot(idx_test, testPredict, color='pink')
# plt.show()

x_indices = [i for i in range(len(idx_test))]

# plt.plot(x_indices, y_test_true, color='blue')
# plt.plot(x_indices, testPredict, color='pink')
# plt.show()
plot_results(testPredict, y_test_true.values)