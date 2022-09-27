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

        # spliting the dataset
        x_train, x_test, y_train, y_test = method.trainValSplit_Kfold(data)

        print(x_train)

        # ... add your methods here in the loop
        # you can add the method's function in ML_Methods object (Like that simple QDA)








