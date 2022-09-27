from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import RepeatedKFold


class ML_Methods:

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

    def trainValSplit_Kfold(self, dataset, num_repeat=1, num_split=10, random_state=None):
        """
        Create dataset and divide dataset to train and test set with number of folding which user has desired.
        Args:
        ---
            `num_repeat` (`int`, optional): How many times this folding should be repeated. Defaults to 1.
            `num_split` (`int`, optional): Number of folding/ spliting dataset. Defaults to 10.
            `random_state` (`random_state`, optional): The state of Randomization. Defaults to None.

        Return: 4 list of datasets which are splited and folded.

        Example for return:
            out = ds.trainValCreation()
            '''
                out[0][0] --> the first train-set
                ...
                out[0][9] --> the tenth train-set

                out[1][0] --> the first test_set
                ...
                out[1][9] --> the tenth test_set

                out[2][0] --> the first train_targets
                ...
                out[2][9] --> the tenth train_targets

                out[3][0] --> the first test_targets
                ...
                out[3][9] --> the tenth test_targets
            '''
        """
        raw_X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values

        scaler = MinMaxScaler()
        X = scaler.fit_transform(raw_X)

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        kf = RepeatedKFold(n_splits=num_split, n_repeats=num_repeat, random_state=random_state)
        for train_index, test_index in kf.split(X):
            # print("Train:", train_index, "Validation:",test_index)
            x_train.append(X[train_index])
            x_test.append(X[test_index])
            y_train.append(Y[train_index])
            y_test.append(Y[test_index])

        print("X_train shape: ".format(np.array(x_train).shape))
        print("X_test shape: ".format(np.array(x_test).shape))
        print("Y_test shape: ".format(np.array(y_train).shape))
        print("Y_test shape: ".format(np.array(y_test).shape))

        return x_train, x_test, y_train, y_test

    def preprocess(self, df):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = MinMaxScaler()
        x = scaler.fit_transform(X)

        print(x.shape)
        print(y.shape)

        return x, y

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
