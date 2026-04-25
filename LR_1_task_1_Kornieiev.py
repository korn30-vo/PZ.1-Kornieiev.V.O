import numpy as np
from sklearn import preprocessing

input_data = np.array([
[5.1, -2.9, 3.3],
[-1.2, 7.8, -6.1],
[3.9, 0.4, 2.1],
[7.3, -9.9, -4.5]
])

# Бінаризація
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("Binarized:")
print(data_binarized)

# Середнє
print("Mean =", input_data.mean(axis=0))
print("Std =", input_data.std(axis=0))

# Масштабування
data_scaled = preprocessing.scale(input_data)
print(data_scaled)

# MinMax
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
print(scaler.fit_transform(input_data))

# Нормалізація
print(preprocessing.normalize(input_data, norm='l1'))
print(preprocessing.normalize(input_data, norm='l2'))
