import pandas as pd
import matplotlib.pyplot as plt


samples = 8352


df = pd.read_csv('ANDR1602_clean.csv', sep=',')
df = df.drop(columns=["id", "time step"])
features = ["wind direction", "temperature", "humidity", "pressure",
            "dewpoint", "wind speed at 2 meters", "solar radiation"]
target = ["wind speed"]

X = [i for i in range(samples)]
y = df[target]

# plt.scatter(X_test, y_test_true, color='red')
# plt.plot(idx_test, y_test_true, color='blue')
plt.plot(X[10:100], y[10:100], color='pink')
plt.show()

