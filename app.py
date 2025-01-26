from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        
        # Check if input is a list of samples
        features_list = data.get('features_list', [])
        if not features_list:
            return jsonify({'error': 'No input data provided!'}), 400
        
        # Convert to numpy array and predict for each sample
        predictions = []
        species = ['setosa', 'versicolor', 'virginica']
        for features in features_list:
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            predicted_species = species[prediction[0]]
            predictions.append(predicted_species)
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
