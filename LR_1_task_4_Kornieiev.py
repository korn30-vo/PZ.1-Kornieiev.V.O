from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([[1,2],[2,3],[3,1],[6,5],[7,8],[8,6]])
y = np.array([0,0,0,1,1,1])

model = GaussianNB()
model.fit(X,y)

pred = model.predict(X)

print(pred)
