import os
import json
import sqlite3
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = 'household_electricity_prediction_secret_key'

DB_PATH = 'predictions.db'
MODEL_PATH = 'models/electricity_model.pkl'
METADATA_PATH = 'models/metadata.json'
CSV_PATH = 'data/household_electricity.csv'

def init_db():
    """Initialize SQLite database to record prediction history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            people INTEGER NOT NULL,
            ac_hours REAL NOT NULL,
            fans INTEGER NOT NULL,
            appliances INTEGER NOT NULL,
            daily_hours REAL NOT NULL,
            temperature REAL NOT NULL,
            prev_consumption REAL NOT NULL,
            predicted_daily REAL NOT NULL,
            estimated_monthly REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_metadata():
    """Load pre-computed model metadata and dataset overview."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return None

def get_recent_predictions(limit=10):
    """Retrieve recent predictions from SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.route('/')
def index():
    metadata = load_metadata()
    history = get_recent_predictions()
    
    # Check if model exists
    model_loaded = os.path.exists(MODEL_PATH)
    
    # Load sample preview if csv exists
    sample_df = None
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            sample_df = df.head(6).to_dict(orient='records')
        except Exception as e:
            print(f"Error loading preview: {e}")

    return render_template(
        'index.html',
        metadata=metadata,
        history=history,
        model_loaded=model_loaded,
        sample_df=sample_df,
        prediction_result=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    metadata = load_metadata()
    history = get_recent_predictions()
    sample_df = None
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            sample_df = df.head(6).to_dict(orient='records')
        except Exception as e:
            pass

    try:
        # 1. Extract input fields
        people = float(request.form.get('people', 0))
        ac_hours = float(request.form.get('ac_usage_hours', 0))
        fans = float(request.form.get('number_of_fans', 0))
        appliances = float(request.form.get('number_of_appliances', 0))
        daily_hours = float(request.form.get('daily_usage_hours', 0))
        temperature = float(request.form.get('outside_temperature', 0))
        prev_consumption = float(request.form.get('previous_consumption', 0))

        # 2. Input Validation (Prevent negative or illogical values)
        errors = []
        if people <= 0 or people > 50:
            errors.append("Number of people must be between 1 and 50.")
        if ac_hours < 0 or ac_hours > 24:
            errors.append("AC usage hours must be between 0 and 24 hours/day.")
        if fans < 0 or fans > 50:
            errors.append("Number of fans must be between 0 and 50.")
        if appliances < 0 or appliances > 100:
            errors.append("Number of appliances must be between 0 and 100.")
        if daily_hours <= 0 or daily_hours > 24:
            errors.append("Daily electricity usage hours must be between 0.1 and 24 hours.")
        if temperature < -20 or temperature > 60:
            errors.append("Outside temperature must be between -20°C and 60°C.")
        if prev_consumption < 0 or prev_consumption > 300:
            errors.append("Previous electricity consumption must be between 0 and 300 kWh/day.")

        if errors:
            for err in errors:
                flash(err, 'danger')
            return render_template(
                'index.html',
                metadata=metadata,
                history=history,
                model_loaded=os.path.exists(MODEL_PATH),
                sample_df=sample_df,
                prediction_result=None,
                input_values=request.form
            )

        # 3. Verify Model Availability
        if not os.path.exists(MODEL_PATH):
            flash("Model file not found! Please run train_model.py first.", 'danger')
            return redirect(url_for('index'))

        # 4. Load trained model and predict
        model = joblib.load(MODEL_PATH)
        
        # Feature order matches training features:
        feature_names = [
            'People', 'AC_Usage_Hours', 'Number_of_Fans', 'Number_of_Appliances', 
            'Daily_Usage_Hours', 'Outside_Temperature', 'Previous_Consumption'
        ]
        input_df = pd.DataFrame(
            [[people, ac_hours, fans, appliances, daily_hours, temperature, prev_consumption]], 
            columns=feature_names
        )
        
        raw_prediction = model.predict(input_df)[0]
        # Electricity consumption cannot be negative in practice
        predicted_daily = max(0.5, float(raw_prediction))
        predicted_daily_rounded = round(predicted_daily, 2)
        
        # 5. Monthly Consumption Estimate
        # Estimated Monthly Consumption = Predicted Daily Consumption * 30
        estimated_monthly_rounded = round(predicted_daily_rounded * 30.0, 2)

        # 6. Save to SQLite History
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, people, ac_hours, fans, appliances, 
                daily_hours, temperature, prev_consumption, 
                predicted_daily, estimated_monthly
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp_str, int(people), round(ac_hours, 1), int(fans), 
            int(appliances), round(daily_hours, 1), round(temperature, 1), 
            round(prev_consumption, 2), predicted_daily_rounded, estimated_monthly_rounded
        ))
        conn.commit()
        conn.close()

        # Update history
        history = get_recent_predictions()

        prediction_result = {
            'daily_kwh': predicted_daily_rounded,
            'monthly_kwh': estimated_monthly_rounded,
            'inputs': {
                'people': int(people),
                'ac_hours': round(ac_hours, 1),
                'fans': int(fans),
                'appliances': int(appliances),
                'daily_hours': round(daily_hours, 1),
                'temperature': round(temperature, 1),
                'prev_consumption': round(prev_consumption, 2)
            }
        }

        flash("Electricity consumption predicted successfully!", "success")
        return render_template(
            'index.html',
            metadata=metadata,
            history=history,
            model_loaded=True,
            sample_df=sample_df,
            prediction_result=prediction_result,
            input_values=request.form,
            scroll_to='prediction-result'
        )

    except Exception as e:
        flash(f"Error during prediction: {str(e)}", "danger")
        return render_template(
            'index.html',
            metadata=metadata,
            history=history,
            model_loaded=os.path.exists(MODEL_PATH),
            sample_df=sample_df,
            prediction_result=None,
            input_values=request.form
        )

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        flash("Prediction history cleared successfully.", "info")
    except Exception as e:
        flash(f"Error clearing history: {e}", "danger")
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"Starting Household Electricity Consumption Prediction Server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
