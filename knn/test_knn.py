import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# loading the iris dataset and storing the data as x and label as y
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 80-20(%) split for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plot to see the dataset
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

# importing our custom knn classifer
from knn_classifier import knn
clf = knn(k=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# checking accuracy
accuracy = np.sum(y_pred == y_test)/len(y_test)
print(accuracy)

