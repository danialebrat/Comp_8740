from Assignment_1.Datasets import Datasets
from Assignment_1.ML_Methods import ML_Methods

# Path to the folder of Datasets
FOLDER_PATH = "C:/Users/User/PycharmProjects/Comp_8740/Assignment_1/Datasets"


if __name__ == "__main__":


    Data = Datasets(folder_path=FOLDER_PATH)
    Data.read_data()


    for name,data in zip(Data.Name_List, Data.Data_List):

        method = ML_Methods(name=name, dataset=data)
        method.preprocess(method.dataset)

        # spliting the datset (not implemented yet)

        # ... add your methods here in the loop
        # you can add the method's function in ML_Methods object (Like that simple QDA)








