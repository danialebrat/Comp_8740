import pip
pip.main(['install', 'tensorflow'])
pip.main(['install', 'openpyxl'])
import os
os.chdir(os.getcwd() + '/')
from ML_Methods import ML_Methods

DATA_SET = ""

# Path to the folder of Datasets
DATASET_PATH = os.getcwd() + "/Datasets/"


if __name__ == "__main__":


    Data = Datasets(folder_path=FOLDER_PATH)
    Data.read_data()


    for name,data in zip(Data.Name_List, Data.Data_List):

        # creating and ML_method object for each dataset
        method = ML_Methods(name=name, dataset=data)
        X, Y = method.preprocess(data)

        # spliting the dataset
        x_train, x_test, y_train, y_test = method.data_spliting(X, Y)

        # Adding methods:
        Models = method.adding_methods()

        # train the models and getting the results
        results, method_name = method.Kfold_report(Models, x_train, y_train, name)

        method.plotting(results, method_name, name)

        method.training_models(Models, x_train, x_test, y_train, y_test, name)

        for model_name, model in Models:

            method.plot_decision_boundary(model, X, Y, model_name, name)