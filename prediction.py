import joblib as jblb
import numpy as np

def predPrint(id):
   model = jblb.load('Iris_model.pkl')
   pred = model.prediction(id)
   str = ["Prediction: ", pred]
   return str
