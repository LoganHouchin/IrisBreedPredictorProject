from joblib import load
import numpy as np

model = load("Iris_model.pkl")

def predPrint(id):
   print("Prediction: " + model.prediction(id))
