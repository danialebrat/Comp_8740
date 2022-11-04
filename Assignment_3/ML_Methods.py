from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neural_network import MLPClassifier


PLOT_PATH = "C:/Users/User/PycharmProjects/Comp_8740/Assignment_1/Assignment_3/Plots/"


class ML_Methods:

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset


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
        Models.append(self.Decision_Tree())
        Models.append(self.Random_Forest())
        Models.append(self.Deep_Neural_Network())

        return Models


    def Kfold_report(self, Models, x_train, y_train, dataset_name):
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
            CrossValidation = cross_val_score(model, x_train, y_train, cv=KFold, scoring='accuracy')
            results.append(CrossValidation)
            method_names.append(name)
            print(f"{name} Training Accuracy : {CrossValidation.mean()*100:.2f}%")

        return results, method_names


    def training_models(self, Models, x_train, x_test, y_train, y_test, datasetname):

        for name, model in Models:
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            cm = confusion_matrix(y_test, predicted)
            AS = accuracy_score(y_test, predicted)

            self.confusion_metrics(cm, AS, name, datasetname)



    def Decision_Tree(self):
        """
        create a Decision_Tree classifier
        :return (name of the mode, Decision_Tree model):
        """
        name = "Desicion_Tree"
        DT_Model = tree.DecisionTreeClassifier()
        return (name, DT_Model)


    def Random_Forest(self):
        """
        create a Random_Forest classifier
        :return (name of the mode, Random_Forest model):
        """
        name = "Random_Forest"
        RF_Model = RandomForestClassifier()
        return (name, RF_Model)


    def Keras_Deep_Neural_Network(self):
        """
        create a Deep_Neural_Network classifier
        """

        DNN_model = Sequential()

        DNN_model.add(Dense(32, activation='relu', input_dim=2))
        DNN_model.add(Dense(16, activation='relu'))
        DNN_model.add(Dense(32, activation='relu'))

        DNN_model.add(Dense(2, activation='sigmoid'))

        DNN_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return DNN_model


    def Deep_Neural_Network(self):
        """
        create a Deep_Neural_Network classifier
        :return (name of the mode, Deep_Neural_Network model):
        """
        name = "Deep_Neural_Network"

        DNN_model = Sequential()

        DNN_model.add(Dense(32, activation='relu', input_dim=2))
        DNN_model.add(Dense(16, activation='relu'))
        DNN_model.add(Dense(32, activation='relu'))

        DNN_model.add(Dense(2, activation='sigmoid'))

        DNN_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Wrap Keras model so it can be used by scikit-learn
        model = KerasClassifier(DNN_model,
                                epochs=20,
                                batch_size=10,
                                verbose=0)

        return (name, model)




    def Neural_Network(self):
        """
        create a Neural_Network classifier
        :return (name of the mode, Neural_Network model):
        """
        name = "Neural_Network"
        NN_model = MLPClassifier(hidden_layer_sizes=(12, 8), activation='relu', max_iter=20, batch_size=10, verbose=False)
        return (name, NN_model)


    def data_spliting(self, x, y, test_size=0.2, random_state=1):
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
        boxplot = plt.boxplot(results, patch_artist=True, labels=names)


        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'lime', 'grey']
        for box, color in zip(boxplot['boxes'], colors):
            box.set(color=color)

        title = "Classifiers Comparison _ {}".format(dataset_name)
        plt.title(title)

        # saving the plots
        fname = PLOT_PATH + title + ".png"
        plt.savefig(fname, dpi=100)
        plt.close('all')


    def confusion_metrics(self, conf_matrix, accuracy_score, method_name, dataset_name):

        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]

        # calculate the sensitivity
        conf_sensitivity = (TP / (float(TP + FN)+ 0.000001))*100
        # calculate the specificity
        conf_specificity = (TN / (float(TN + FP) + 0.000001))*100
        # calculate PPV
        ppv = (TP / (float(TP + FP) + 0.000001))*100
        # calculate NPV
        npv = (TN / (float(TN + FN) + 0.000001))*100

        print("**************")
        print("Classifier: {} _ Dataset: {}".format(method_name, dataset_name))
        print("PPV:{:.2f} NPV:{:.2f} Sensitivity:{:.2f} Specificity:{:.2f}".format(ppv, npv, conf_sensitivity, conf_specificity))
        print("Accuracy Score for test_set: {:.2f} ".format(accuracy_score*100))

    def plot_decision_boundary(self, model, X, Y, model_name, dataset_name):

        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        x_in = np.c_[xx.ravel(), yy.ravel()]

        # Predict the function value for the whole gid
        y_pred = model.predict(x_in)
        y_pred = np.round(y_pred).reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, y_pred, cmap="Pastel1")
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap="Pastel2")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())


        title = "Decision boundry of {} on {}". format(model_name, dataset_name)
        plt.title(title)

        # saving the plots
        fname = PLOT_PATH + title + ".png"
        plt.savefig(fname, dpi=100)
        plt.close('all')



