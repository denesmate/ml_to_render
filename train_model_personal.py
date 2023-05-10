# Script to train machine learning model.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import joblib

PTH = "./census.csv"

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Add code to load in the data.

def import_data(PTH):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data = pd.read_csv(PTH, skipinitialspace = True)
    return data

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# Proces the test data with the process_data function.

def process_data(data, random_state_number):
    X = pd.DataFrame(data, columns=cat_features)
    y = pd.DataFrame(data, columns=['salary'])
    # Split the data into train and validation, stratifying on the target feature.
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=random_state_number)

    #for later validation
    X_test = X_val.reset_index(drop=True).copy()

    X_train_describe = X_train.describe()

    return X_train, X_val, X_test, y_train, y_val, X_train_describe

# Train and save a model.

def train_data(X_train, X_val, X_test, y_train, y_val, X_train_describe):
    lr = LogisticRegression(max_iter=1000, random_state=23)
    lb = LabelBinarizer()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_train = ohe.fit_transform(X_train.values)
    X_val = ohe.transform(X_val.values)

    # Binarize the target feature.
    y_train = lb.fit_transform(y_train)
    y_val = lb.transform(y_val)

    # Train Logistic Regression.
    lr.fit(X_train, y_train.ravel())

    # the function outputs the performance on slices of just the categorical features

    output_PTH = '../model/'

    for i in range(len(X_train_describe.loc["top"])):
        column_name = X_train_describe.columns[i]
        top_value = X_train_describe.loc["top"][i]
        row_slice = X_test[column_name] == top_value
        print(f"F1 score on {top_value} {column_name} slices:")
        score = f1_score(y_val[row_slice], lr.predict(X_val[row_slice]))
        print(score)

    score = f1_score(y_val[row_slice], lr.predict(X_val[row_slice]))

    joblib.dump(lr, output_PTH + "/logistic_model.pkl")

    return score

# X_train, y_train, encoder, lb = process_data(
#     train, categorical_features=cat_features, label="salary", training=True
# )


if __name__ == "__main__":
    data = import_data(PTH)
    X_train, X_val, X_test, y_train, y_val, X_train_describe = process_data(data, random_state_number = 23)
    train_data(X_train, X_val, X_test, y_train, y_val, X_train_describe)
