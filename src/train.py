import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

def split_data(df):
    train = df[df['Year']<2021]
    val = df[df['Year']==2022]
    test = df[df['Year']>=2023]

    return train, val, test

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse,y_pred

def save_model(model, file_path):
    joblib.dump(model, file_path)

def save_predictions(df,preds,path="data/predictions/predictions_script.csv"):
    results = df.copy()
    results['predictions'] = preds
    results.to_csv(path,index=False)