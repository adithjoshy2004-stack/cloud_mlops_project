from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Iris Flower Prediction API",
    description="Machine Learning Prediction API",
    version="1.0"
)

# Load the trained model
model = joblib.load("model/iris_model_v1.pkl")


# Input data
class FlowerData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris ML API is running",
        "model_version": "v1"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: FlowerData):

    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)

    classes = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    result = classes[int(prediction[0])]

    return {
        "prediction": result,
        "model_version": "v1"
    }