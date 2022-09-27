from Assignment_1.Datasets import Datasets
from Assignment_1.ML_Methods import ML_Methods

# Path to the folder of Datasets
FOLDER_PATH = "C:/Users/User/PycharmProjects/Comp_8740/Assignment_1/Assignment_1/Datasets"


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

        # train the models
        method.Train_Models(Models, x_train, y_train, name)


        #performing methods for each fold
        #for x_train, y_train, x_test, y_test in zip(x_trains, x_tests, y_trains, y_tests):
        # you can add the method's function in ML_Methods object (Like that simple QDA)








