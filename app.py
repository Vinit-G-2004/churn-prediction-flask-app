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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature1 = float(request.form.get('feature1'))  # Using get() to avoid KeyError
        feature2 = float(request.form.get('feature2'))
        feature3 = float(request.form.get('feature3'))

        # You can add additional checks to ensure features are correctly passed
        # Example: If a feature is missing, handle it gracefully
        if feature1 is None or feature2 is None or feature3 is None:
            return "Missing required feature values", 400
        
        # Process the features and make prediction (Your existing logic goes here)
        # result = model.predict([feature1, feature2, feature3])
        # return result

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

