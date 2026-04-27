import joblib
import pandas as pd

def load_model(file_path):
    return joblib.load(file_path)

def predict(model,input_data:dict):
    df = pd.DataFrame([input_data])
    predictions = model.predict(df)
    return predictions[0]

