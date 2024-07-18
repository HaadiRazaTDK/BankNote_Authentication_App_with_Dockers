from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

# Load the classifier
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict', methods=['GET'])
def predict_note_authentication():
    """Let's Authenticate the Banks Note
    This is using docstring for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
      200:
        description: The prediction of bank note authentication
    """
    # Get parameters from request
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    
    # Check if all parameters are provided
    if variance is None or skewness is None or curtosis is None or entropy is None:
        return jsonify({'error': 'Missing parameter(s)'}), 400
                
    # Convert parameters to float
    variance = float(variance)
    skewness = float(skewness)
    curtosis = float(curtosis)
    entropy = float(entropy)
                
    # Prepare the feature vector for prediction
    features = np.array([[variance, skewness, curtosis, entropy]])
                
    # Make prediction
    prediction = classifier.predict(features)
                
    # Return the prediction result
    return jsonify({'prediction': int(prediction[0])})
            

@app.route('/predict_file', methods=['POST'])
def predict_note_file():
    """Let's authenticate the Banks Note
    This is using the docstring for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: The output values
    """
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
        
    # Return the prediction result
    return "The predicted values for the csv is " + str(list(prediction))


if __name__ == '__main__':
    app.run(debug=True)
