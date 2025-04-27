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
    # Get input data from the form
    features = [float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4']),
                float(request.form['feature5']),
                float(request.form['feature6']),
                float(request.form['feature7'])]
    
    # Reshape the input data to match model's expected input format
    features_array = np.array(features).reshape(1, -1)

    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(features_array)

    # Make a prediction using the model
    prediction = model.predict(scaled_data)
    
    # Convert the prediction into a readable result (e.g., 0 = No Churn, 1 = Churn)
    result = 'Churn' if prediction[0] == 1 else 'No Churn'

    # Render the result page with the prediction
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

