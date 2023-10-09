import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv('poker-hand-training-true.data', sep=',',header=None)

print(data.head())



dataList = data[10].to_list()
sorted_list = sorted(dataList)
sorted_counted = Counter(sorted_list)

range_length = list(range(max(dataList))) 
data_series = {}

for i in range_length:
    data_series[i] = 0 

for key, value in sorted_counted.items():
    data_series[key] = value

data_series = pd.Series(data_series)
x_values = data_series.index


plt.xlim(0, max(dataList))
plt.title("Distribution of Categories (Numerical)")
plt.xlabel("Category")
plt.ylabel("Values")
plt.scatter(x_values, data_series.values)
plt.show() 

#for some reason the values were appearing incorrectly on the pie chart when seting the labels
reverse_x_values = x_values[::-1]
plt.xlim(0, max(dataList))
plt.title("Distribution of Categories (Percentage)")
plt.pie(x_values, labels=reverse_x_values)
plt.show() 