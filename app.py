"""
Iris Flower Classifier - Flask Web Application
Serves the web dashboard and exposes the prediction API powered
by the custom NumPy Multi-Layer Perceptron (MLP) trained from Iris.ipynb.
"""

import os
from flask import Flask, render_template, request, jsonify
from model import model

app = Flask(__name__)

# Input boundary constraints for validation
INPUT_CONSTRAINTS = {
    'sepal_length': {'min': 3.5, 'max': 9.0, 'name': 'Sepal Length (cm)'},
    'sepal_width':  {'min': 1.5, 'max': 5.0, 'name': 'Sepal Width (cm)'},
    'petal_length': {'min': 0.5, 'max': 8.0, 'name': 'Petal Length (cm)'},
    'petal_width':  {'min': 0.05, 'max': 3.5, 'name': 'Petal Width (cm)'}
}


@app.route('/')
def index():
    """Render the main classifier web page."""
    metrics_info = model.get_metrics_and_info()
    return render_template('index.html', info=metrics_info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction API endpoint.
    Expects JSON: {sepal_length, sepal_width, petal_length, petal_width}
    """
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Request payload must be a valid JSON object.'
        }), 400

    data = request.get_json()

    # Validate presence of required features
    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return jsonify({
            'success': False,
            'error': f"Missing required measurements: {', '.join(missing_fields)}"
        }), 400

    # Validate data types and reasonable ranges
    parsed_values = {}
    for field in required_fields:
        val = data.get(field)
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': f"Measurement '{field}' must be a valid number."
            }), 400

        constraint = INPUT_CONSTRAINTS[field]
        if val_float < constraint['min'] or val_float > constraint['max']:
            return jsonify({
                'success': False,
                'error': f"{constraint['name']} must be between {constraint['min']} and {constraint['max']} cm."
            }), 400

        parsed_values[field] = val_float

    try:
        # Run inference using the trained NumPy MLP
        result = model.predict(
            sepal_length=parsed_values['sepal_length'],
            sepal_width=parsed_values['sepal_width'],
            petal_length=parsed_values['petal_length'],
            petal_width=parsed_values['petal_width']
        )

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'image_url': result['image_url'],
            'inputs': parsed_values
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'An internal error occurred during model prediction.'
        }), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Return model architecture, dataset properties, and verified performance metrics."""
    metrics_info = model.get_metrics_and_info()
    if metrics_info:
        return jsonify({'success': True, 'data': metrics_info})
    return jsonify({'success': False, 'error': 'Metrics not found.'}), 404


if __name__ == '__main__':
    # Default local development port 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
