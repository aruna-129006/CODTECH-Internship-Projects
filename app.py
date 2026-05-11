from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():

    # Get input value
    hours = float(request.form['hours'])

    # Predict marks
    prediction = model.predict([[hours]])

    # Show result
    return render_template(
        "index.html",
        prediction_text=f"Predicted Marks: {prediction[0]:.2f}"
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
