from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

        x_trains = []
        x_tests = []
        y_trains = []
        y_tests = []

        kf = RepeatedKFold(n_splits=num_split, n_repeats=num_repeat, random_state=random_state, shuffle=True)
        for train_index, test_index in kf.split(X):
            # print("Train:", train_index, "Validation:",test_index)
            x_trains.append(X[train_index])
            x_tests.append(X[test_index])
            y_trains.append(Y[train_index])
            y_tests.append(Y[test_index])

        return x_trains, x_tests, y_trains, y_tests

    def preprocess(self, df):
        """
        create x and y from a pandas dataframe
        x, which are 2D point will be scaled using min-max scaler

        :param dataframe:
        :return (Scaled X (minmax), y):
        """

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        scaler = MinMaxScaler()
        x = scaler.fit_transform(X)

        return x, y

    def adding_methods(self):
        """
        adding all the methods with their specific names in a list

        :return: a List containing tuple of models (name of the model, model)
        """

        Models = []

        # models
        Models.append(self.QDA())
        Models.append(self.LDA())

        return Models


    def Train_Models(self, Models, x_train, y_train, dataset_name):
        """
        training all the models from the list of models using 10 fold cross validation

        :param x_train:
        :param y_train:
        :return:
        """

        print("**********")
        print("{} Dataset Results: ".format(dataset_name))

        results = []
        method_names = []
        for name, model in Models:
            # train the models
            KFold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            CrossValidation = cross_val_score(model, x_train, y_train, cv=KFold, scoring="accuracy")
            results.append(CrossValidation)
            method_names.append(name)
            print(f"{name} Accuracy : {CrossValidation.mean()*100:.2f}%")

        return results, method_names

    def QDA(self):
        """
        create a quadratic-discriminant-analysis classifier
        :return (name of the mode, QDA model):
        """
        name = "QDA"
        QDA_model = QuadraticDiscriminantAnalysis()
        return (name , QDA_model)


    def LDA(self):
        """
        create a linear-discriminant-analysis classifier
        :return (name of the mode, QDA model):
        """
        name = "LDA"
        clf = LinearDiscriminantAnalysis()
        return (name, clf)


    def data_spliting(self, x, y, test_size=0.1, random_state=1):
        """
        Split the data into x_train, x_test, y_train, y_test

        :param x: x (data)
        :param y: y (labels)
        :param test_size: size of test dataset
        :param random_state: 1 or 0
        :return: x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test


    def plotting(self, results, names, dataset_name):

        plt.figure(figsize=(12, 10))
        plt.boxplot(results, labels=names)
        plt.title("Classifiers Comparison _ {}".format(dataset_name))
        plt.show()
