from typing import Counter
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.MultiColumnEncoder import MultiColumnLabelEncoder
from imblearn.under_sampling import ClusterCentroids
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def oversample(data: pd.DataFrame):

        # Drop null values if any
        data.dropna(inplace=True)

        # Get features and labels (X, y)
        y = data[["stroke"]].copy()
        X = data.copy()
        X.drop(columns= ["id", "stroke"], inplace=True)

        # Bring input in right format
        X_input = X.values # input must be 2d array
        y_input = y.values.flatten() # should be 1d array

        # Initialize SMOTENC oversampler (the numbers are the columns that contain categorical values)
        smote = SMOTENC(categorical_features=[0, 2, 3, 4, 5, 6, 9], random_state=0) 
        # fit predictor and target variable
        x_smote, y_smote = smote.fit_resample(X_input, y_input)

        print(f'Original dataset samples per class {Counter(y_input)}')
        print(f'Resampled dataset samples per class {Counter(y_smote)}')

        # Output back as Dataframe
        X_output = pd.DataFrame(x_smote)
        X_output.set_axis(["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"], axis=1, inplace=True)
        X_output["stroke"] = y_smote

        # Datatypes are changed from int to float after oversampling, need to change it back 
        float_int = ["age", "hypertension", "heart_disease"]
        for col in float_int:
            X_output[col] = X_output[col].astype(int)

        return X_output


def undersample_kmeans(data: pd.DataFrame):

    y_under = data[["stroke"]].copy()
    X_under = data.copy()
    X_under.drop(columns= ["id", "stroke"], inplace=True)

   # Based on KMEANS -> we need to apply one hot encoding

    # First encode all categorical columns
    onehot_encoder = OneHotEncoder()

    categorical = ["hypertension", "heart_disease", "gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    other = ["age", "avg_glucose_level", "bmi"]

    X_onehot = onehot_encoder.fit_transform(X_under[categorical])

    other_as_array = np.array(X_under[other])
    X_transformend = np.concatenate((X_onehot.toarray(), other_as_array), axis=1)

    X_input = X_transformend # input must be 2d array
    y_input = y_under.values.flatten() # should be 1d array

    # Then proceed with undersamping 
    cc = ClusterCentroids(random_state=0)

    X_kmeans_resampled, y_kmeans_resampled = cc.fit_resample(X_input, y_input)

    from typing import Counter
    print(f'Original dataset samples per class {Counter(y_input)}')
    print(f'Resampled dataset samples per class {Counter(y_kmeans_resampled)}')

    # reverse one hot encoding
    reverse = onehot_encoder.inverse_transform(X_kmeans_resampled[:,:20])
    X_combined = np.concatenate((X_kmeans_resampled[:,20:25], reverse), axis=1)

    # Return as dataframe 
    X_output = pd.DataFrame(X_combined)
    X_output
    X_output.set_axis(["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "gender", "ever_married", "work_type", "Residence_type", "smoking_status"], axis=1, inplace=True)
    X_output["stroke"] = y_kmeans_resampled

    #  Datatypes are changed from int to float after undersampling, need to change it back 
    float_int = ["age"]
    for col in float_int:
        X_output[col] = X_output[col].astype(int)
    
    return X_output