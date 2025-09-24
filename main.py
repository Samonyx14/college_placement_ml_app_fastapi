from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
import kagglehub
import os
from typing import Tuple
import joblib

# define a funtion that loads the data from kaggle
def get_data() -> pd.DataFrame:
    # Download latest version
    path = kagglehub.dataset_download("vrajesh0sharma7/college-student-placement")
    data_path = os.listdir(path)
    data_url = os.path.join(path, data_path[0])
    final_data = pd.read_csv(data_url)
    final_data.to_csv('spotify.csv')
    
    return final_data
def encode_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # drop user_id
    if "College_ID" in list(data.columns):
        data.drop(columns=["College_ID"], inplace=True)
    # encode the categorical variables
    label_encoders = {}
    cat_cols = list(data.select_dtypes(include="object").columns)
    for col in cat_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        label_encoders[col] = encoder
        
    # serialize the encoder
    joblib.dump(value=label_encoders, filename="label_encoders.pkl")
    
    return data, label_encoders
def split_scale_data(data:pd.DataFrame) -> Tuple:
    X_train, X_test, y_train, y_test = None, None, None, None
    try:
        X, y = data.drop(columns=['Placement']), data['Placement']
        X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                            test_size=0.2, random_state=23,
                                                            stratify=y)
# scale the dataset
        scaler = StandardScaler()
        column_names = list(X_train.columns)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train = pd.DataFrame(data=X_train_scaled, columns=column_names)
        X_test = pd.DataFrame(data=X_test_scaled, columns=column_names)
        
        joblib.dump(value=scaler, filename="scaler.pkl")
    except Exception as e:
        print(f"error detail: {e}")
    
    return X_train, X_test, y_train, y_test
# function that performs training

def training(X_train:pd.DataFrame, X_test:pd.DataFrame,
             y_train:pd.Series, y_test:pd.Series) -> Tuple:
    model = RandomForestClassifier(n_estimators=5, random_state=23)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_score = f1_score(y_train, train_preds)
    test_score = f1_score(y_test, test_preds)
    
    # serialize
    joblib.dump(value=model, filename="model.pkl")
    
    return train_score, test_score
if __name__ == "__main__":
    data = get_data()
    encoded_data, encoder = encode_dataset(data)
    X_train, X_test, y_train, y_test = split_scale_data(data=encoded_data)
    train_score, test_score = training(X_train, X_test, y_train, y_test)
    print(train_score)
    print(test_score)