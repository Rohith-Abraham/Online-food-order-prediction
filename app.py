from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the model using the correct load function from joblib
model = load("D:\\sju\\CDS\\Aman\\project\\project\\trained_model.joblib")

@app.route('/', methods=['GET', 'POST'])
def home():  # Renamed the function to avoid naming conflict
    if request.method == 'POST':
        try:
            # Get values from the form
            age = float(request.form['age'])
            gender = request.form['gender']
            marital_status = request.form['marital_status']
            occupation = request.form['occupation']
            monthly_income = float(request.form['monthly_income'])
            family_size = float(request.form['family_size'])
            feedback = float(request.form['feedback'])

            # Create a list or array with the feature values
            features = [[age, gender, marital_status, occupation, monthly_income, family_size, feedback]]

            # Make a prediction using the trained model
            prediction = model.predict(features)

            # Render the result template with the prediction
            return render_template('result.html', prediction=prediction)

        except Exception as e:
            return render_template('index.html', error=str(e))  # Display error if occurs

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
