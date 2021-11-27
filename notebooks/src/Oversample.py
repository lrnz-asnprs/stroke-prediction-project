from typing import Counter
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.Multicolumnencoder import MultiColumnLabelEncoder

def oversample(data: pd.DataFrame):

        # Drop null values if any
        data.dropna(inplace=True)

        # Get features and labels (X, y)
        y = data[["stroke"]].copy()
        X = data.copy()
        X.drop(columns= ["id", "stroke"], inplace=True)

        # First encode all categorical columns
        categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
        multi = MultiColumnLabelEncoder(columns=categorical)
        X = multi.fit_transform(X)

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
        float_int = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]
        for col in float_int:
            X_output[col] = X_output[col].astype(int)

        # Now the encoded categorical values can be changed back to initial values
        inv = multi.inverse_transform(X_output)

        return inv


