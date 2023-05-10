import pandas as pd
import numpy as np
from train_model_personal import train_data, process_data, import_data
from sklearn.preprocessing import OneHotEncoder
import os
import logging


# csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = pd.read_csv(csv_url, header = None)

#  testing if any ML functions return the expected type.


data = import_data("./census.csv")
X_train, X_val, X_test, y_train, y_val, X_train_describe = process_data(data)

def test_import_data(data):
    try:
        assert isinstance(data, pd.DataFrame)
        print(" - data type is DataFrame")
    except AssertionError as err:
        print(" - Error: data type is not DataFrame")
        pass
    

def test_process_data(data):
    X_train = process_data(data)
    try:
        assert type(X_train) is tuple
        print(" - X_train type is tuple")
    except AssertionError as err:
        print(" - Error: X_train type is not tuple")
        pass
    
def test_train_data(X_train):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_train_encoded = ohe.fit_transform(X_train.values)
    try:
        assert type(X_train_encoded)is np.ndarray
        print(" - The one-hot encoded X_train variable type is array")
    except AssertionError as err:
        print(" - Error: The one-hot encoded X_train variable type is not an array")
        raise err

# def test_f1_score_type():
#     assert(train_data != 0)


# def test_f1_score_type(train_data):
#     PTH = "./census.csv"
#     data = import_data(PTH)
#     X_train, X_val, X_test, y_train, y_val, X_train_describe = process_data(data)
#     score = train_data(X_train, X_val, X_test, y_train, y_val, X_train_describe)
#     assert type(train_data.score) is float, "F1 score value is float type"

# def test_train_models(train_data):
#     assert os.path.exists("../model/logistic_model.pkl") is True


if __name__ == "__main__":
    #test_f1_score_type(cls.train_data)
    #test_train_models(cls.train_data)
    #test_f1_score_type()
    test_import_data(data)
    test_process_data(data)
    test_train_data(X_train)