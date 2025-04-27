from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    # Serve the home page where the user will enter input data
    return render_template('index.html')

# Define your prediction function
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        print(f"Received inputs: {feature1}, {feature2}, {feature3}")  # Logging inputs for debugging

        # Create a 2D array with the input features
        input_data = [[feature1, feature2, feature3]]

        # Apply scaling (if you used StandardScaler or any other scaler during training)
        scaled_input = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(scaled_input)

        # Log the prediction result
        print(f"Prediction result: {prediction}")

        # Return the prediction result (you can modify this to display it on your HTML page)
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

