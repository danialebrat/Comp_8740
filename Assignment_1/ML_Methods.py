from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class ML_Methods:


    def __init__(self, name, dataset):

        self.name = name
        self.dataset = dataset


    def preprocess(self, df):

        data = []
        points = []
        labels = []

        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        label = df.iloc[:, 2].values

        for i, j, k in zip(x, y, label):
            points.append([i, j])
            labels.append([k])

        arr = np.array(points).reshape(-1,1)
        scaler = MinMaxScaler()
        new_points = scaler.fit_transform(arr)


        for i, j in zip(new_points, labels):
            data.append([i, j])

        print(np.array(data).shape)
        return np.array(data)


    def QDA(self, x_train, y_train):

        QDA_model = QuadraticDiscriminantAnalysis()
        QDA_model.fit(x_train, y_train)
        return QDA_model






# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
# clf = QuadraticDiscriminantAnalysis()
# clf.fit(X, y)
# QuadraticDiscriminantAnalysis()
# print(clf.predict([[-0.8, -1]]))