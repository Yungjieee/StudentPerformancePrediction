from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the Lasso model and scaler
with open('best_lasso_model.pkl', 'rb') as model_file:
    lasso_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define route for the form
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [
        float(request.form['study_hours_per_week']),
        float(request.form['attendance_rate']),
        float(request.form['previous_exam_scores']),
        float(request.form['assignments_completed']),
        float(request.form['extracurricular_participation']),
    ]
    # Generate engineered features
    study_hours_per_week = features[0]
    attendance_rate = features[1]
    study_attendance_interaction = study_hours_per_week * attendance_rate
    study_hours_per_week_squared = study_hours_per_week ** 2
    attendance_rate_squared = attendance_rate ** 2
    features += [study_attendance_interaction, study_hours_per_week_squared, attendance_rate_squared]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict with the Lasso model
    prediction = lasso_model.predict(features_scaled)[0]

    # Set a maximum score of 100
    if prediction > 100:
        prediction = 100

    # Render the template with prediction
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
