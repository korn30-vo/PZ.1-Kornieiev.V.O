import numpy as np
from sklearn import preprocessing

input_data = np.array([
[-3.3, -1.6, 6.1],
[2.4, -1.2, 4.3],
[-3.2, 5.5, -6.1],
[-4.4, 1.4, -1.2]
])

# Бінаризація
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print(data_binarized)

# Масштабування
print(preprocessing.scale(input_data))

# MinMax
scaler = preprocessing.MinMaxScaler()
print(scaler.fit_transform(input_data))

# Нормалізація
print(preprocessing.normalize(input_data, norm='l1'))
print(preprocessing.normalize(input_data, norm='l2'))
