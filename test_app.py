"""
Automated Test Suite for Iris Flower Classifier Website & Model API
"""

import unittest
import json
from app import app
from model import model


class TestIrisClassifier(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_direct_predictions(self):
        """Verify model predictions across all 3 classes."""
        # Test Setosa
        res_setosa = model.predict(5.1, 3.5, 1.4, 0.2)
        self.assertEqual(res_setosa['prediction'], 'Iris-setosa')
        self.assertGreater(res_setosa['confidence'], 90.0)
        self.assertIn('Iris-setosa', res_setosa['probabilities'])

        # Test Versicolor
        res_versicolor = model.predict(5.7, 2.8, 4.1, 1.3)
        self.assertEqual(res_versicolor['prediction'], 'Iris-versicolor')
        self.assertGreater(res_versicolor['confidence'], 50.0)

        # Test Virginica
        res_virginica = model.predict(6.5, 3.0, 5.5, 1.8)
        self.assertEqual(res_virginica['prediction'], 'Iris-virginica')
        self.assertGreater(res_virginica['confidence'], 50.0)

    def test_index_route(self):
        """Verify index route renders HTML successfully."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('Iris Flower Classifier', html)
        self.assertIn('Multi-Layer Perceptron (MLP) from Scratch', html)
        self.assertIn('Enter Flower Measurements', html)
        self.assertIn('Prediction Result', html)
        self.assertIn('UCI Iris Dataset', html)

    def test_api_info_route(self):
        """Verify GET /api/info returns complete project metadata."""
        response = self.app.get('/api/info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['model']['architecture'], '4-8-3')
        self.assertEqual(data['data']['performance']['accuracy'], '96.67%')

    def test_predict_endpoint_valid(self):
        """Verify POST /predict returns valid JSON with prediction and probabilities."""
        payload = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 'Iris-setosa')
        self.assertIn('confidence', data)
        self.assertIn('probabilities', data)
        self.assertIn('image_url', data)

    def test_predict_endpoint_missing_fields(self):
        """Verify POST /predict handles missing fields with HTTP 400."""
        payload = {'sepal_length': 5.1}
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Missing required measurements', data['error'])

    def test_predict_endpoint_out_of_bounds(self):
        """Verify POST /predict validates range bounds with HTTP 400."""
        payload = {
            'sepal_length': 25.0,  # invalid bound
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('must be between', data['error'])

    def test_predict_endpoint_non_numeric(self):
        """Verify POST /predict rejects non-numeric input with HTTP 400."""
        payload = {
            'sepal_length': 'invalid_string',
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.app.post('/predict',
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])


if __name__ == '__main__':
    unittest.main()
