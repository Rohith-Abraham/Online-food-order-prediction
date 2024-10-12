import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump  # You can use joblib to save and load the model
from sklearn.model_selection import train_test_split
####
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
####

data=pd.read_csv("D:\SJC\CDS\onlinefoods1.csv")
###
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Marital Status"] = data["Marital Status"].map({"Married": 2, 
                                                     "Single": 1, 
                                                     "Prefer not to say": 0})
data["Occupation"] = data["Occupation"].map({"Student": 1, 
                                             "Employee": 2, 
                                             "Self Employeed": 3, 
                                             "House wife": 4})
data["Educational Qualifications"] = data["Educational Qualifications"].map({"Graduate": 1, 
                                                                             "Post Graduate": 2, 
                                                                             "Ph.D": 3, "School": 4, 
                                                                             "Uneducated": 5})
data["Monthly Income"] = data["Monthly Income"].map({"No Income": 0, 
                                                     "25001 to 50000": 5000, 
                                                     "More than 50000": 7000, 
                                                     "10001 to 25000": 25000, 
                                                     "Below Rs.10000": 10000})
data["Feedback"] = data["Feedback"].map({"Positive": 1, "Negative ": 0})
###
x = np.array(data[["Age", "Gender", "Marital Status", "Occupation", "Monthly Income", "Family size", "Feedback"]])
y = np.array(data["Output"])


# Assuming you have 'xtrain' and 'ytrain' for training
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# Save the trained model to a file
dump(model, 'trained_model.joblib')
from joblib import load
model = load('trained_model.joblib')