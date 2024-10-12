import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Assuming you've already trained your model and stored it in the 'model' variable

with open('model.pkl','rb+') as file:
    model= pickle.load(file)

def predict(age, gender, marital_status, occupation, monthly_income, family_size, feedback):
    # Prepare user input as a numpy array
    user_input = np.array([[age, gender, marital_status, occupation, monthly_income, family_size, feedback]])
    
    # Use your trained model to make predictions
    prediction = model.predict(age, gender, marital_status, occupation, monthly_income, family_size, feedback)

    # 'model.predict' will return an array; extract the first prediction
    prediction = prediction[0]

    return prediction
