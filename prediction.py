from joblib import load
import numpy as np

model = ("Iris_Breed_Predictor.iypnb")

def predPrint(id):
   print("Prediction: " + model.prediction(id))
