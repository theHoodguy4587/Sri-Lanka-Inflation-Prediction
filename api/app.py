from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model_bundle = joblib.load("models/model_script.joblib")
model = model_bundle['model']
model_columns = model_bundle['columns']


class PredictionInput(BaseModel):
    Country: str
    GDP_Growth: float
    Inflation_GDP_Deflator: float
    GDP_per_Capita: float
    Inflation_Lag1: float
    Inflation_Lag2: float
    GDP_Growth_Lag1: float

@app.post("/predict")
def predict(input_data: PredictionInput):

    
    """
    Example input_data:
    {
        "Country": "Sri Lanka",
        "GDP_Growth": 3.5,
        "Inflation_GDP_Deflator": 4.2,
        "GDP_per_Capita": 4000,
        "Inflation_Lag1": 5.2,
        "Inflation_Lag2": 4.8,
        "GDP_Growth_Lag1": 3.0
    }
    """
    df = pd.DataFrame([input_data.model_dump()])

    df = pd.get_dummies(df, columns=["Country"])

    for col in model_columns:
        if col not in df:
            df[col] = 0

    df = df[model_columns]

    prediction = model.predict(df)

    return {"predicted_inflation":float(prediction[0])}