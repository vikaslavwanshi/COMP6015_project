# backend/app.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('random_forest_model.pkl')

def predict_attack(raw_data_file, test_data_file):
    # Load and preprocess your data
    raw_data = pd.read_csv(raw_data_file)
    test_data = pd.read_csv(test_data_file)

    # Drop the datetime column from the test data
    test_data = test_data.drop(columns=['Timestamp'])
    
    # Assume the test_data contains the features needed for prediction
    features = test_data.drop(columns=['Normal/Attack'])
    
    # Make predictions
    predictions = model.predict(features)
    
    # Return the results as a DataFrame or any other suitable format
    result_df = pd.DataFrame({'predictions': predictions})
    return result_df

@app.route('/')
def home():
    return "Welcome to the Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data_file = request.files['raw_data_file']
        test_data_file = request.files['test_data_file']
        app.logger.debug(f'Received files: {raw_data_file.filename}, {test_data_file.filename}')
        
        # Process the files and make predictions
        result = predict_attack(raw_data_file, test_data_file)
        result_json = result.to_json(orient='records')
        return jsonify(result_json)
    except Exception as e:
        app.logger.error(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
