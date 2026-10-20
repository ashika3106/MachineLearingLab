import os
import unittest
import numpy as np
import joblib
from app import app, init_db, get_db_connection, DB_PATH, MODEL_PATH

class TestElectricityApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        init_db()

    def test_model_file_exists_and_loads(self):
        """Verify the trained model exists and can perform inference."""
        self.assertTrue(os.path.exists(MODEL_PATH), "Model file models/electricity_model.pkl must exist.")
        model = joblib.load(MODEL_PATH)
        self.assertIsNotNone(model)
        
        # Test prediction with example: People=4, AC=5, Fans=3, Appliances=8, Daily=10, Temp=34, Prev=10
        feature_names = [
            'People', 'AC_Usage_Hours', 'Number_of_Fans', 'Number_of_Appliances', 
            'Daily_Usage_Hours', 'Outside_Temperature', 'Previous_Consumption'
        ]
        import pandas as pd
        test_input = pd.DataFrame([[4, 5.0, 3, 8, 10.0, 34.0, 10.0]], columns=feature_names)
        pred = model.predict(test_input)[0]
        self.assertIsInstance(pred, (float, np.floating))
        self.assertGreater(pred, 0, "Predicted electricity consumption must be positive.")
        print(f"\n[Test] Model test prediction for 4 people, 5h AC, 3 fans, 8 appliances, 34°C: {pred:.2f} kWh/day")

    def test_get_home_page(self):
        """Verify the home page loads with 200 OK and contains key figures/sections."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Household Electricity Consumption Prediction', html)
        self.assertIn('Fig. 1', html)
        self.assertIn('Fig. 2', html)
        self.assertIn('Fig. 3', html)
        self.assertIn('Fig. 4', html)
        self.assertIn('Fig. 5', html)
        self.assertIn('Fig. 6', html)
        self.assertIn('consumption_dist.png', html)
        self.assertIn('ac_vs_consumption.png', html)
        self.assertIn('temp_vs_consumption.png', html)
        self.assertIn('prev_vs_actual.png', html)
        self.assertIn('correlation_heatmap.png', html)
        self.assertIn('actual_vs_predicted.png', html)

    def test_prediction_post_success(self):
        """Verify submitting the form produces valid daily and monthly predictions."""
        form_data = {
            'people': '4',
            'ac_usage_hours': '5.0',
            'number_of_fans': '3',
            'number_of_appliances': '8',
            'daily_usage_hours': '10.0',
            'outside_temperature': '34.0',
            'previous_consumption': '10.0'
        }
        response = self.client.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Predicted Daily Consumption', html)
        self.assertIn('Estimated Monthly Consumption', html)
        self.assertIn('kWh/day', html)
        self.assertIn('kWh/month', html)

    def test_prediction_post_invalid_inputs(self):
        """Verify validation prevents negative or extreme values."""
        invalid_data = {
            'people': '-2',
            'ac_usage_hours': '35.0', # > 24 hrs
            'number_of_fans': '3',
            'number_of_appliances': '8',
            'daily_usage_hours': '10.0',
            'outside_temperature': '34.0',
            'previous_consumption': '10.0'
        }
        response = self.client.post('/predict', data=invalid_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Number of people must be between 1 and 50', html)
        self.assertIn('AC usage hours must be between 0 and 24', html)

    def test_sqlite_history_and_clear(self):
        """Verify SQLite prediction logging and clearing."""
        form_data = {
            'people': '2',
            'ac_usage_hours': '2.0',
            'number_of_fans': '2',
            'number_of_appliances': '4',
            'daily_usage_hours': '8.0',
            'outside_temperature': '28.0',
            'previous_consumption': '7.5'
        }
        self.client.post('/predict', data=form_data, follow_redirects=True)
        
        # Verify row in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM predictions')
        rows = cursor.fetchall()
        conn.close()
        self.assertGreater(len(rows), 0)

        # Clear history
        clear_resp = self.client.post('/clear-history', follow_redirects=True)
        self.assertEqual(clear_resp.status_code, 200)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM predictions')
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

if __name__ == '__main__':
    unittest.main()
