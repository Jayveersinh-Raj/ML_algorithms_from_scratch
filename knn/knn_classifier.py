import numpy as np
from collections import Counter


# euclidean distance class
# general formula for n dimensions = sq.root of summation till i = 1 to n (x-xi)^2
def euclidean_distance(x, X):
    return np.sqrt(np.sum((x - X)**2))



# knn class
class knn:
    # default configuration with k = 3
    def __init__(self, k=3):
        self.k = k

    # fit method
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # prediction method to call predict_each method and store predictions
    def predict(self, x):
        y_pred = [self.predict_each(i) for i in x]
        return np.array(y_pred) # returning it as an array

    # prediction for each by finding the distance 
    def predict_each(self, x):
        distances = [euclidean_distance(x, i) for i in self.X_train]

        # k nearest class labels (indices)
        k_nearest_indices = np.argsort(distances)[:self.k]

        # k nearest class labels from the indices
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
         
        # getting the most common of them to give it as the predicted value
        most_common_label = Counter(k_nearest_labels).most_common(1) # 1 most common
        return most_common_label[0][0] # because most_common will give tuple with the most common,
                                       # and its frequency so we want the label value thus [0][0]
