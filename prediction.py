from joblib import load
import numpy as np

iris_data = datasets.load_iris()
class_names = iris_data.target_names

def predPrint(id):
   model = load('Iris_model.pkl')
   dummy = np.array(id)
   Dummy = dummy.reshape(1,-1)
   prediction = model.predict(Dummy)
   Pred = class_names[prediction]
   str = ["Prediction: ",Pred]
   return str
