import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import norm
import statistics
import seaborn as sns
import sys
filename = 'Data/Data_for_project.csv'
df = pd.read_csv(filename)
np.set_printoptions(threshold=np.inf)

attributeNames = df.columns[1:-2].tolist()

raw_data = df.values
X = raw_data[:, range(1, 10)]
y = raw_data[:, 10]
N = X.shape[0]
M = X.shape[1]

for i in range(0, N):
    X[i][4] = 1.0 if X[i][4] == "Present" else 0.0

X = X.astype(float)

standard = zscore(X, ddof=1)

mean = [0 for i in range(len(raw_data[0]))]
std = [0 for i in range(len(raw_data[0]))]
median = [0 for i in range(len(raw_data[0]))]
rang = [0 for i in range(len(raw_data[0]))]
min = [0 for i in range(len(raw_data[0]))]
max = [0 for i in range(len(raw_data[0]))]

for i in range(len(X[0])):
    mean[i] = X[:, i].mean()
    std[i] = X[:, i].std(ddof=1)
    median[i] = np.median(X[:, i])
    rang[i] = X[:, i].max()-X[:, i].min()
    min[i] = X[:, i].min()
    max[i] = X[:, i].max()
    print("Name: {}, Min: {} Max: {}".format(attributeNames[i], round(min[i],2), round(max[i],2)))
    print("Mean: {}, std: {}, median: {}, range: {} for {}".format(round(mean[i],2), round(std[i],2), round(median[i],2), round(rang[i],2), attributeNames[i]))

dataframe = pd.DataFrame(standard,
                         columns=attributeNames)
matrix = dataframe.corr()
matrix.style.background_gradient(cmap='coolwarm')
print(matrix.to_string())

sns.heatmap(matrix, annot=True, fmt=".2f")
plt.title("Correlation matrix")
plt.show()


plt.figure(figsize=(14,9))

u = np.floor(np.sqrt(M))
v = np.ceil(float(M)/u)
plt.style.use('seaborn-whitegrid')
bins = 40
for col in range(len(X[0])):
    plt.subplot(int(u), int(v), col + 1)
    mu, std = norm.fit(X[:, col])
    plt.hist(X[:, col], bins=bins, facecolor='#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.axvline(X[:, col].mean(), color='k', linestyle='dashed', linewidth=1)
    plt.plot(x, p * len(X) * (xmax - xmin) / bins, 'k', linewidth=2)
    plt.xlabel(attributeNames[col])
    plt.ylabel('Occurrence')
plt.suptitle('Histogram of attributes with normal distribution')
plt.show()
