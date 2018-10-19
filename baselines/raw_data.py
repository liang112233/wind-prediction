import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

samples = 8352


df = pd.read_csv('ANDR1602_clean.csv', sep=',')
# df = df.drop(columns=["id", "time step"])
features = ["wind direction", "temperature", "humidity", "pressure",
            "dewpoint", "wind speed at 2 meters", "solar radiation"]
target = ["wind speed"]

X = [i for i in range(samples)]
features_df = df[features]
y = df[target]

# plt.scatter(X_test, y_test_true, color='red')
# plt.plot(idx_test, y_test_true, color='blue')
# plt.plot(X[10:100], y[10:100], color='pink')
# plt.show()

#### correlation calculation ###
for f_index, f in enumerate(features):
    my_feature = features_df.values[:, f_index]
    my_target = y.values[:, 0]
    coeff, pvalue = pearsonr(my_feature, my_target)
    print("feature:", f, ", yield:", target[0], "; coeff:", coeff, ";p-value:", pvalue)

# yield_year_count = len(labels.columns)
# for f in range(features):
#     for y in range(yield_year_count):
#         f_label = df.columns[8+f]
#         y_label = df.columns[3 + y]
#         coeff, pvalue = pearsonr(dat[:, f], labels.values[:, y])
#         print("feature:", f_label, ", yield:", y_label, "; coeff:", coeff, ";p-value:", pvalue)