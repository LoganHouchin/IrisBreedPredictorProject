from joblib import load
import numpy as np

def predPrint(id):
   model = load('Iris_model.pkl')
   dummy = np.array(id)
   Dummy = dummy.reshape(1,-1)
   prediction = model.pred(Dummy)
   str = ["Prediction: ",prediction]
   return str
