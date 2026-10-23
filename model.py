"""
NumPy-based Multi-Layer Perceptron (MLP) Model Inference Engine
Reuses the trained weights, biases, and standard scaler parameters
fitted from the UCI Iris dataset and Iris.ipynb.
"""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
WEIGHTS_FILE = os.path.join(ARTIFACTS_DIR, 'model_weights.npz')
SCALER_FILE = os.path.join(ARTIFACTS_DIR, 'scaler_params.json')
METRICS_FILE = os.path.join(ARTIFACTS_DIR, 'metrics.json')


class IrisModel:
    def __init__(self):
        self._ensure_artifacts_exist()
        self._load_artifacts()

    def _ensure_artifacts_exist(self):
        """Train and generate artifacts if they are not yet present."""
        if not (os.path.exists(WEIGHTS_FILE) and os.path.exists(SCALER_FILE)):
            print("Artifacts missing, running training pipeline...")
            import train_model
            train_model.main()

    def _load_artifacts(self):
        """Load trained weights, biases, scaler parameters, and class names."""
        # Load weights
        weights_data = np.load(WEIGHTS_FILE)
        self.W1 = weights_data['W1']  # Shape (4, 8)
        self.b1 = weights_data['b1']  # Shape (1, 8)
        self.W2 = weights_data['W2']  # Shape (8, 3)
        self.b2 = weights_data['b2']  # Shape (1, 3)

        # Load scaler parameters & class labels
        with open(SCALER_FILE, 'r', encoding='utf-8') as f:
            scaler_data = json.load(f)

        self.feature_names = scaler_data['feature_names']
        self.scaler_mean = np.array(scaler_data['mean'], dtype=np.float64)
        self.scaler_scale = np.array(scaler_data['scale'], dtype=np.float64)
        self.classes = scaler_data['classes']

        # Species image mapping
        self.image_map = {
            'Iris-setosa': '/static/images/iris_setosa.jpg',
            'Iris-versicolor': '/static/images/iris_versicolor.jpg',
            'Iris-virginica': '/static/images/iris_virginica.jpg'
        }

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """Forward pass through the 4-8-3 MLP network."""
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return A2

    def predict(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
        """
        Standardizes the 4 flower measurements and computes the predicted
        species, confidence score, and class probabilities.
        """
        raw_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=np.float64)
        
        # Standardize using fitted mean and scale
        scaled_features = (raw_features - self.scaler_mean) / self.scaler_scale

        # Forward pass
        probabilities = self.forward_propagation(scaled_features)[0]

        # Predicted class index and name
        predicted_idx = int(np.argmax(probabilities))
        predicted_species = self.classes[predicted_idx]
        confidence = float(probabilities[predicted_idx] * 100)

        # Probabilities dictionary for all 3 classes
        prob_dict = {
            cls_name: round(float(prob * 100), 2)
            for cls_name, prob in zip(self.classes, probabilities)
        }

        return {
            'prediction': predicted_species,
            'confidence': round(confidence, 2),
            'probabilities': prob_dict,
            'image_url': self.image_map.get(predicted_species, '/static/images/iris_setosa.jpg')
        }

    def get_metrics_and_info(self):
        """Returns verified project metadata, architecture, and performance."""
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


# Global singleton instance
model = IrisModel()
