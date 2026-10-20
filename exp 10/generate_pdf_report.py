import os
import pypdf
from PIL import Image
from playwright.sync_api import sync_playwright

OUTPUT_HTML = "report.html"
OUTPUT_PDF = "Household_Electricity_Consumption_Prediction_Report.pdf"

base_dir = os.path.abspath(".")
img_dir = os.path.join(base_dir, "report_screenshots")

def get_img_path(rel_path):
    p = os.path.join(base_dir, rel_path)
    return p.replace("\\", "/")

def ensure_eda_composite():
    p1 = os.path.join(base_dir, 'static/plots/consumption_dist.png')
    p2 = os.path.join(base_dir, 'static/plots/ac_vs_consumption.png')
    out_p = os.path.join(base_dir, 'report_screenshots/fig4_eda_composite.png')
    
    if os.path.exists(p1) and os.path.exists(p2):
        img1 = Image.open(p1)
        img2 = Image.open(p2)
        h = min(img1.height, img2.height)
        w1 = int(img1.width * (h / img1.height))
        w2 = int(img2.width * (h / img2.height))
        img1_r = img1.resize((w1, h), Image.Resampling.LANCZOS)
        img2_r = img2.resize((w2, h), Image.Resampling.LANCZOS)
        composite = Image.new('RGB', (w1 + w2 + 24, h), (255, 255, 255))
        composite.paste(img1_r, (0, 0))
        composite.paste(img2_r, (w1 + 24, 0))
        composite.save(out_p)

def generate_report():
    ensure_eda_composite()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Household Electricity Consumption Prediction Using Machine Learning</title>
<style>
    @page {{
        size: A4 portrait;
        margin: 18mm 20mm 18mm 20mm;
    }}
    
    * {{
        box-sizing: border-box;
    }}

    body {{
        font-family: "Times New Roman", Times, Georgia, serif;
        font-size: 10.5pt;
        line-height: 1.38;
        color: #111111;
        margin: 0;
        padding: 0;
        background: #ffffff;
    }}

    .page {{
        width: 100%;
        page-break-after: always;
        position: relative;
    }}

    .page:last-child {{
        page-break-after: avoid;
    }}

    /* Typography */
    .doc-title {{
        text-align: center;
        font-size: 13.5pt;
        font-weight: bold;
        margin-top: 5px;
        margin-bottom: 3px;
        line-height: 1.25;
    }}

    .doc-subtitle {{
        text-align: center;
        font-size: 10.5pt;
        font-style: italic;
        margin-bottom: 18px;
    }}

    .section-title {{
        font-size: 11pt;
        font-weight: bold;
        margin-top: 14px;
        margin-bottom: 6px;
    }}

    p {{
        margin-top: 0;
        margin-bottom: 8px;
        text-align: justify;
    }}

    ol, ul {{
        margin-top: 4px;
        margin-bottom: 10px;
        padding-left: 24px;
    }}

    li {{
        margin-bottom: 4px;
        text-align: justify;
    }}

    /* Code Block Box */
    .code-box {{
        background-color: #f8f9fa;
        border: 1px solid #d1d5db;
        border-radius: 2px;
        padding: 8px 10px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8pt;
        line-height: 1.3;
        color: #24292e;
        white-space: pre-wrap;
        word-break: break-word;
        margin-bottom: 10px;
    }}

    .code-title {{
        font-size: 10pt;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 4px;
    }}

    .code-subtext {{
        font-size: 9.5pt;
        font-style: italic;
        color: #333333;
        margin-bottom: 8px;
    }}

    /* Figure styling */
    .fig-container {{
        text-align: center;
        margin-top: 8px;
        margin-bottom: 12px;
    }}

    .fig-img {{
        max-width: 92%;
        height: auto;
        border: 1px solid #e5e7eb;
        border-radius: 3px;
        display: block;
        margin: 0 auto 5px auto;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    .fig-img-vscode {{
        max-width: 360px;
        max-height: 480px;
        object-fit: contain;
    }}

    .fig-img-med {{
        max-width: 92%;
        max-height: 250px;
        object-fit: contain;
    }}

    .fig-img-wide {{
        max-width: 95%;
        max-height: 250px;
        object-fit: contain;
    }}

    .fig-caption {{
        text-align: center;
        font-size: 9.5pt;
        font-style: italic;
        color: #222222;
        margin-top: 3px;
        margin-bottom: 10px;
        line-height: 1.25;
    }}
</style>
</head>
<body>

<!-- ================= PAGE 1 ================= -->
<div class="page">
    <div class="doc-title">Household Electricity Consumption Prediction Using Machine Learning</div>
    <div class="doc-subtitle">(Daily &amp; Monthly Electricity Consumption Prediction using Regression Algorithms)</div>

    <div class="section-title">Problem Statement:</div>
    <p>
        Residential electricity consumption represents a substantial proportion of global energy demand and is subject to significant financial and environmental implications. In contemporary households, daily electricity usage fluctuates dynamically based on household occupancy, seasonal weather patterns, appliance ownership, and intensive cooling loads such as air conditioning. Without data-driven forecasting mechanisms, consumers frequently face unexpectedly high electricity bills, while municipal energy grids struggle with demand spikes and inefficient load scheduling.
    </p>
    <p>
        Manual calculations or static heuristics are inherently incapable of capturing the multi-dimensional and non-linear interactions between demographic variables, ambient outdoor temperatures, appliance counts, and baseline consumption history. There is, therefore, a pressing requirement for an automated, fast, transparent, and accessible system capable of accurately predicting daily electricity consumption in kilowatt-hours per day (kWh/day) and computing estimated monthly usage (kWh/month) from readily observable household inputs.
    </p>
    <p>
        This project designs, trains, and implements an end-to-end web-based Household Electricity Consumption Prediction system using supervised machine learning regression algorithms. The system performs data preprocessing, exploratory data analysis (EDA), multi-model training and comparison (Linear Regression, Decision Tree Regressor, Random Forest Regressor), hyperparameter configuration, model persistence via Joblib, and live inference through an interactive Flask web interface. The trained models are systematically evaluated using standard regression metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination (R&sup2; Score).
    </p>

    <div class="section-title">Objectives:</div>
    <ol>
        <li>To collect and preprocess a structured residential dataset capturing core household attributes (occupants, AC usage hours, fan counts, appliances, daily usage hours, ambient temperature, and previous consumption).</li>
        <li>To perform exploratory data analysis (EDA) and generate visual distribution plots, feature trendlines, and correlation heatmaps illustrating energy dynamics.</li>
        <li>To design, train, and compare multiple regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor) using an 80:20 train-test split.</li>
        <li>To evaluate the candidate models on an unseen test set and select the optimal model based on lowest test Root Mean Squared Error (RMSE) and highest R&sup2; score.</li>
        <li>To serialize and persist the best-performing model using Joblib and build a responsive Flask-based web application for real-time inference.</li>
        <li>To provide real-time daily predictions (kWh/day), calculate estimated monthly consumption (kWh/month), and persist prediction history with timestamps in an SQLite database.</li>
    </ol>

    <div class="section-title">Software / Hardware Required:</div>
    <ul>
        <li><strong>Operating System:</strong> Windows / Linux / macOS</li>
        <li><strong>IDE:</strong> Visual Studio Code</li>
        <li><strong>Language:</strong> Python 3.x</li>
        <li><strong>Backend Framework:</strong> Flask (REST API &amp; MVC Architecture)</li>
    </ul>
</div>

<!-- ================= PAGE 2 ================= -->
<div class="page">
    <ul>
        <li><strong>Machine Learning:</strong> scikit-learn (LinearRegression, DecisionTreeRegressor, RandomForestRegressor)</li>
        <li><strong>Data Processing &amp; Visualization:</strong> Pandas, NumPy, Matplotlib, Seaborn</li>
        <li><strong>Database:</strong> SQLite3 (Local prediction history log)</li>
        <li><strong>Model Persistence:</strong> Joblib</li>
        <li><strong>Frontend:</strong> HTML5, CSS3 (Modern Light Theme), JavaScript</li>
        <li><strong>Dataset:</strong> Household Electricity Consumption Dataset (600 records, 8 attributes)</li>
        <li><strong>Version Control:</strong> Git</li>
    </ul>

    <div class="section-title">Project Directory Structure:</div>
    <p>
        The project (&ldquo;HouseholdElectricityPrediction&rdquo;) is organized as a clean Flask backend with dedicated data, model, static visual assets, and template directories, as shown below.
    </p>

    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig1_vscode_structure.png')}" class="fig-img fig-img-vscode" alt="Project Directory Structure in VS Code">
        <div class="fig-caption">Fig 1: Project directory structure of HouseholdElectricityPrediction in Visual Studio Code</div>
    </div>
</div>

<!-- ================= PAGE 3 ================= -->
<div class="page">
    <div class="section-title">Procedure:</div>
    <ol>
        <li>Collect and organize a realistic dataset (600 records) capturing 7 input features and the target daily electricity consumption in <code>data/household_electricity.csv</code>.</li>
        <li>Run data preprocessing: verify zero null values, detect and eliminate duplicate rows, and cast columns to appropriate numerical formats (<code>int64</code>, <code>float64</code>).</li>
        <li>Execute Exploratory Data Analysis (EDA) to produce Matplotlib/Seaborn visualizations: target consumption histogram, AC vs. kWh trendline, outdoor temperature vs. kWh, previous vs. actual kWh, and the correlation heatmap.</li>
        <li>Partition the preprocessed dataset into training (80%, 480 samples) and testing (20%, 120 samples) sets using <code>random_state=42</code> for strict reproducibility.</li>
        <li>Train multiple regression algorithms: Ordinary Least Squares (Linear Regression), Decision Tree Regressor, and Random Forest Regressor on the training partition.</li>
        <li>Evaluate all candidate models on the unseen test set using MAE, MSE, RMSE, and R&sup2; Score, and select the optimal model with the lowest test RMSE.</li>
        <li>Serialize the selected best model to <code>models/electricity_model.pkl</code> and export metadata to <code>models/metadata.json</code> using Joblib.</li>
        <li>Launch the Flask web application (<code>app.py</code>) and access the user interface via local browser at <code>http://127.0.0.1:5050</code>.</li>
        <li>Accept household inputs via the web form with input boundary validation preventing negative or out-of-range inputs.</li>
        <li>Feed validated inputs to the serialized model, output the predicted daily consumption (kWh/day), calculate the estimated monthly consumption (Daily &times; 30), and record each query to the SQLite database (<code>predictions.db</code>).</li>
    </ol>

    <div class="section-title">Implementation Code:</div>
    <div class="code-subtext">Representative implementation of the key modules used in the project, based on the pipeline shown in the application (adapt variable/file names to match your exact source files).</div>

    <div class="code-title">1. Data Preprocessing &amp; Pipeline Setup (train_model.py)</div>
    <div class="code-box">import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset and verify integrity
df = pd.read_csv('data/household_electricity.csv')
print(f"Dataset shape: {{df.shape}} | Missing values: {{df.isnull().sum().sum()}}")

features = ['People', 'AC_Usage_Hours', 'Number_of_Fans', 'Number_of_Appliances', 
            'Daily_Usage_Hours', 'Outside_Temperature', 'Previous_Consumption']
target = 'Electricity_Consumption'

X = df[features]
y = df[target]

# 80:20 Train-Test Partition with fixed random seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train samples: {{len(X_train)}} | Test samples: {{len(X_test)}}")</div>
</div>

<!-- ================= PAGE 4 ================= -->
<div class="page">
    <div class="code-title">2. Exploratory Data Analysis &amp; Visualization (train_model.py)</div>
    <div class="code-box">import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Target Consumption Distribution
plt.figure(figsize=(7, 4.5))
sns.histplot(df['Electricity_Consumption'], kde=True, color='#2563eb', bins=25)
plt.title('Distribution of Daily Electricity Consumption', fontweight='bold')
plt.xlabel('Daily Electricity Consumption (kWh/day)')
plt.savefig('static/plots/consumption_dist.png', dpi=150)
plt.close()

# 2. AC Usage vs. Electricity Consumption
plt.figure(figsize=(7, 4.5))
sns.regplot(data=df, x='AC_Usage_Hours', y='Electricity_Consumption',
            scatter_kws={{'alpha': 0.5, 'color': '#0284c7'}}, line_kws={{'color': '#e11d48'}})
plt.title('AC Usage Hours vs Electricity Consumption', fontweight='bold')
plt.savefig('static/plots/ac_vs_consumption.png', dpi=150)
plt.close()

# 3. Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig('static/plots/correlation_heatmap.png', dpi=150)
plt.close()</div>

    <div class="code-title">3. Model Training, Comparison &amp; Serialization (train_model.py)</div>
    <div class="code-box">from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

models = {{
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
}}

results = {{}}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {{
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }}

# Select best model with lowest RMSE
best_model_name = min(results.keys(), key=lambda k: results[k]['RMSE'])
best_model = models[best_model_name]
joblib.dump(best_model, 'models/electricity_model.pkl')
print(f"Selected Best Model: {{best_model_name}} (Saved to models/electricity_model.pkl)")</div>
</div>

<!-- ================= PAGE 5 ================= -->
<div class="page">
    <div class="code-title">4. Prediction Route &amp; Estimation Backend (app.py)</div>
    <div class="code-box">from flask import Flask, render_template, request, flash
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'models/electricity_model.pkl'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate input values
        people = float(request.form.get('people', 0))
        ac_hours = float(request.form.get('ac_usage_hours', 0))
        fans = float(request.form.get('number_of_fans', 0))
        appliances = float(request.form.get('number_of_appliances', 0))
        daily_hours = float(request.form.get('daily_usage_hours', 0))
        temp = float(request.form.get('outside_temperature', 0))
        prev_kwh = float(request.form.get('previous_consumption', 0))

        # Input boundary validation
        if people <= 0 or ac_hours < 0 or daily_hours <= 0:
            flash("Invalid input values. Please provide positive numbers.", "danger")
            return render_template('index.html')

        # Load serialized model
        model = joblib.load(MODEL_PATH)
        feature_names = ['People', 'AC_Usage_Hours', 'Number_of_Fans', 'Number_of_Appliances',
                         'Daily_Usage_Hours', 'Outside_Temperature', 'Previous_Consumption']
        input_df = pd.DataFrame([[people, ac_hours, fans, appliances, daily_hours, temp, prev_kwh]],
                                columns=feature_names)

        # Model inference
        raw_pred = model.predict(input_df)[0]
        predicted_daily = max(0.5, float(raw_pred))
        predicted_daily_rounded = round(predicted_daily, 2)

        # Monthly Consumption Estimate (Daily x 30)
        estimated_monthly_rounded = round(predicted_daily_rounded * 30.0, 2)

        return render_template('index.html',
                               prediction_result={{
                                   'daily_kwh': predicted_daily_rounded,
                                   'monthly_kwh': estimated_monthly_rounded
                               }})
    except Exception as e:
        flash(f"Error during prediction: {{e}}", "danger")
        return render_template('index.html')</div>
</div>

<!-- ================= PAGE 6 ================= -->
<div class="page">
    <div class="code-title">5. Saving Prediction History to SQLite (app.py)</div>
    <div class="code-box">import sqlite3
from datetime import datetime

def log_prediction(people, ac_hours, fans, appliances, daily_hours, temp, prev_kwh, daily_pred, monthly_est):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        people INTEGER, ac_hours REAL, fans INTEGER, appliances INTEGER,
        daily_hours REAL, temperature REAL, prev_consumption REAL,
        predicted_daily REAL, estimated_monthly REAL
    )''')
    
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''INSERT INTO predictions (
        timestamp, people, ac_hours, fans, appliances,
        daily_hours, temperature, prev_consumption,
        predicted_daily, estimated_monthly
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        timestamp_str, int(people), round(ac_hours, 1), int(fans),
        int(appliances), round(daily_hours, 1), round(temp, 1),
        round(prev_kwh, 2), daily_pred, monthly_est
    ))
    conn.commit()
    conn.close()</div>

    <div class="section-title">Output (Step-by-Step Screenshots):</div>
    
    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig2_dataset_overview.png')}" class="fig-img fig-img-med" alt="Dataset Overview">
        <div class="fig-caption">Fig 2: Step 1 &mdash; Dataset Overview: dataset size (600 records), 7 input features, and sample records preview</div>
    </div>
</div>

<!-- ================= PAGE 7 ================= -->
<div class="page">
    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig3_preprocessing.png')}" class="fig-img fig-img-med" alt="Data Preprocessing">
        <div class="fig-caption">Fig 3: Step 2 &mdash; Data Preprocessing: missing-value verification, data types, and 80:20 train-test split configuration</div>
    </div>

    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig4_eda_composite.png')}" class="fig-img fig-img-wide" alt="Exploratory Data Analysis">
        <div class="fig-caption">Fig 4: Step 3 &mdash; Exploratory Data Analysis: daily electricity consumption distribution and AC usage hours impact</div>
    </div>
</div>

<!-- ================= PAGE 8 ================= -->
<div class="page">
    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig5_model_comparison.png')}" class="fig-img fig-img-med" alt="Model Training and Comparison">
        <div class="fig-caption">Fig 5: Step 4 &mdash; Model Training &amp; Comparison: Linear Regression vs. Decision Tree vs. Random Forest Regressor on unseen test set</div>
    </div>

    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig6_evaluation.png')}" class="fig-img fig-img-med" alt="Selected Model Evaluation">
        <div class="fig-caption">Fig 6: Step 5 &mdash; Selected Model Evaluation: MAE (0.6101), MSE (0.5792), RMSE (0.7610), R&sup2; (0.9881), and Actual vs. Predicted scatter plot</div>
    </div>
</div>

<!-- ================= PAGE 9 ================= -->
<div class="page">
    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig7_predict_form.png')}" class="fig-img fig-img-med" alt="Electricity Consumption Prediction Form">
        <div class="fig-caption">Fig 7: Step 6 &mdash; Electricity Consumption Prediction: interactive input form with 7 household and environmental features</div>
    </div>

    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig8_prediction_result_sample1.png')}" class="fig-img fig-img-med" alt="Prediction Output Sample 1">
        <div class="fig-caption">Fig 8: Step 6 &mdash; Prediction output (Test Case 1): Predicted Daily Consumption (14.30 kWh/day) and Estimated Monthly Consumption (429.00 kWh/month)</div>
    </div>
</div>

<!-- ================= PAGE 10 ================= -->
<div class="page">
    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig9_prediction_result_sample2.png')}" class="fig-img fig-img-med" alt="Prediction Output Sample 2">
        <div class="fig-caption">Fig 9: Step 6 &mdash; Prediction output (Test Case 2 - High Summer Demand): Predicted Daily Consumption (31.80 kWh/day) and Monthly Estimate (954.00 kWh/month)</div>
    </div>

    <div class="fig-container">
        <img src="{get_img_path('report_screenshots/fig10_history_log.png')}" class="fig-img fig-img-med" alt="Prediction History Log">
        <div class="fig-caption">Fig 10: Step 7 &mdash; Prediction history log persisted in the SQLite database (predictions.db)</div>
    </div>
</div>

<!-- ================= PAGE 11 ================= -->
<div class="page">
    <div class="fig-container">
        <img src="{get_img_path('static/plots/actual_vs_predicted.png')}" class="fig-img fig-img-med" alt="Actual vs Predicted Detailed Chart">
        <div class="fig-caption">Fig 11: Step 8 &mdash; Actual vs. Predicted Electricity Consumption scatter plot with ideal identity diagonal (y = x)</div>
    </div>

    <div class="section-title">Result:</div>
    <p>
        A web-based Household Electricity Consumption Prediction system was successfully designed, trained, evaluated, and implemented using a Flask web backend, Scikit-learn regression algorithms, and an SQLite database. The dataset comprising 600 records across 7 core electrical and environmental features was preprocessed, verified for zero missing values and zero duplicate records, and partitioned using an 80:20 train-test split (random_state=42).
    </p>
    <p>
        Among the evaluated regression algorithms, Linear Regression demonstrated superior generalization on the unseen test set, achieving a Root Mean Squared Error (RMSE) of 0.7610 kWh/day, a Mean Absolute Error (MAE) of 0.6101 kWh/day, a Mean Squared Error (MSE) of 0.5792 (kWh)&sup2;, and a Coefficient of Determination (R&sup2;) of 0.9881 (explaining 98.81% of total consumption variance). The model was serialized using Joblib and deployed within a modern, responsive web application. The web interface enables users to submit household parameters, produces real-time daily consumption predictions along with estimated monthly totals, and logs every query into an SQLite database for auditing and longitudinal energy management.
    </p>
</div>

</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated intermediate HTML at {OUTPUT_HTML}")

    # Use Playwright to render to PDF with exact A4 print settings
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.goto("file://" + os.path.abspath(OUTPUT_HTML), wait_until="networkidle")
        page.wait_for_timeout(1000)

        margin_dict = {
            "top": "15mm",
            "bottom": "15mm",
            "left": "18mm",
            "right": "18mm"
        }
        page.pdf(
            path=OUTPUT_PDF,
            format="A4",
            print_background=True,
            margin=margin_dict
        )
        browser.close()

    print(f"Successfully generated PDF: {OUTPUT_PDF}")

    # Verify page count with pypdf
    reader = pypdf.PdfReader(OUTPUT_PDF)
    print(f">> Total Page Count in Generated PDF: {len(reader.pages)} pages")

if __name__ == "__main__":
    generate_report()
