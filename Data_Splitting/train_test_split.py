import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from Data_Acquisition.Data_loader import GetData


class SeparateIndependentFeature:

    """

    class Name  : SeperateIndependentFeature
    Description : This class is used to split the dataset in training and testing set.
    Written by  : Adityaraj Hemant Chaudhari
    Version     : 0.1
    Revision    : None

    """

    def __init__(self):
        self.data = GetData()

    def x_y_feat(self):

        """

        Method_Name : x_y_feat
        Description : Splitting the dataset into dependent and independent features.
        Output      : DataFrame
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            data = self.data.acquire_data()
            x = data.drop("Type_Of_Kernel", axis=1)
            y = data["Type_Of_Kernel"]
            return x,y

        except Exception as e:
            raise e

    def scalar(self):

        """

        Method_Name : scalar
        Description : Scaling the dataset.
        Output      : DataFrame
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x, y = self.x_y_feat()
            rob = RobustScaler()
            x_scaled = pd.DataFrame(rob.fit_transform(x), columns=x.columns)
            return x_scaled, y, rob
        except Exception as e:
            raise e

    def save_scalar(self):

        """

        Method_Name : save_scalar
        Description : Saving the scalar in serialized manner.
        Output      : Pkl_File
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            a, b, rob = self.scalar()
            pickle.dump(rob, open('./Scalar.pkl', 'wb'))
        except Exception as e:
            raise e

    def train_test_set(self):

        """

        Method_Name : train_test_set
        Description : This function divides the dataset into training and testing set.
        Output      : DataFrame
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            x, y, rob = self.scalar()
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise e




