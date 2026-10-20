import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating images
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

def generate_synthetic_dataset(filename='data/household_electricity.csv', n_samples=600):
    """
    Generate a realistic synthetic dataset for Household Electricity Consumption Prediction.
    Clearly documented as synthetic data generated with realistic domain relationships.
    """
    print(f"Generating synthetic dataset with {n_samples} records...")
    
    # 1. Number of people in the house (1 to 8)
    people = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_samples, p=[0.10, 0.25, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02])
    
    # 2. Outside Temperature in Celsius (18°C to 44°C)
    outside_temp = np.random.uniform(18.0, 43.0, size=n_samples)
    
    # 3. AC Usage Hours per Day (0 to 16 hours, higher probability of AC use when hotter)
    temp_factor = np.clip((outside_temp - 24.0) / 18.0, 0.0, 1.0)
    ac_usage = np.random.exponential(scale=2.5, size=n_samples) + (temp_factor * 6.0)
    # Some houses have no AC usage (especially on cooler days)
    zero_ac_mask = (outside_temp < 25.0) & (np.random.rand(n_samples) < 0.4)
    ac_usage[zero_ac_mask] = 0.0
    ac_usage = np.clip(ac_usage, 0.0, 16.0)
    
    # 4. Number of Fans (1 to 8, correlated with people and rooms)
    fans = np.clip(people + np.random.choice([-1, 0, 1, 2], size=n_samples, p=[0.2, 0.4, 0.3, 0.1]), 1, 8)
    
    # 5. Number of Electrical Appliances (2 to 16)
    appliances = np.clip(people * 2 + np.random.randint(0, 5, size=n_samples), 2, 16)
    
    # 6. Daily Electricity Usage Hours (4 to 20 hours)
    daily_usage_hours = np.clip(6.0 + (people * 1.2) + np.random.normal(0, 2.0, size=n_samples), 4.0, 20.0)
    
    # Base physics-grounded electricity consumption formula:
    # Base load (fridge, standby, baseline lighting): ~2.0 kWh
    # People load: ~0.8 kWh per person
    # AC power: ~1.4 kWh per hour of AC use, increasing with outside temp > 30C
    # Fan power: ~0.07 kW * 8 hours per fan ~= 0.5 kWh per fan
    # Appliance power: ~0.45 kWh per appliance
    # Daily usage duration multiplier
    
    ac_load = ac_usage * (1.35 + np.maximum(0, (outside_temp - 30.0) * 0.03))
    fan_load = fans * 0.45 * (daily_usage_hours / 12.0)
    people_load = people * 0.85
    appliance_load = appliances * 0.40 * (daily_usage_hours / 14.0)
    base_load = 2.2 + (daily_usage_hours * 0.15)
    
    # Total true daily consumption (kWh)
    calculated_consumption = base_load + people_load + ac_load + fan_load + appliance_load
    
    # Realistic previous consumption (yesterday's reading with slight day-to-day variance)
    prev_consumption = calculated_consumption + np.random.normal(0, 1.2, size=n_samples)
    prev_consumption = np.clip(prev_consumption, 3.5, 55.0)
    
    # Today's electricity consumption incorporates previous consumption momentum + daily variations + noise
    consumption = (0.35 * prev_consumption) + (0.65 * calculated_consumption) + np.random.normal(0, 0.65, size=n_samples)
    consumption = np.clip(consumption, 3.0, 60.0)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'People': people.astype(int),
        'AC_Usage_Hours': np.round(ac_usage, 1),
        'Number_of_Fans': fans.astype(int),
        'Number_of_Appliances': appliances.astype(int),
        'Daily_Usage_Hours': np.round(daily_usage_hours, 1),
        'Outside_Temperature': np.round(outside_temp, 1),
        'Previous_Consumption': np.round(prev_consumption, 2),
        'Electricity_Consumption': np.round(consumption, 2)
    })
    
    df.to_csv(filename, index=False)
    print(f"Dataset successfully saved to {filename} ({len(df)} rows).")
    return df

def generate_eda_plots(df):
    """
    Generate all required EDA plots and save to static/plots/
    """
    print("Generating EDA visualizations...")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 11})

    # 1. Electricity Consumption Distribution
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df['Electricity_Consumption'], kde=True, color='#2563eb', bins=25, edgecolor='black', alpha=0.6)
    plt.title('Distribution of Daily Electricity Consumption', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Daily Electricity Consumption (kWh/day)', fontsize=11)
    plt.ylabel('Household Frequency', fontsize=11)
    mean_val = df['Electricity_Consumption'].mean()
    median_val = df['Electricity_Consumption'].median()
    plt.axvline(mean_val, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f} kWh')
    plt.axvline(median_val, color='#16a34a', linestyle=':', linewidth=1.8, label=f'Median: {median_val:.2f} kWh')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('static/plots/consumption_dist.png', dpi=150)
    plt.close()

    # 2. AC Usage Hours vs Electricity Consumption
    plt.figure(figsize=(7, 4.5))
    sns.regplot(data=df, x='AC_Usage_Hours', y='Electricity_Consumption', 
                scatter_kws={'alpha': 0.5, 'color': '#0284c7', 's': 30},
                line_kws={'color': '#e11d48', 'linewidth': 2})
    plt.title('AC Usage Hours vs Electricity Consumption', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('AC Usage (Hours/day)', fontsize=11)
    plt.ylabel('Electricity Consumption (kWh/day)', fontsize=11)
    plt.tight_layout()
    plt.savefig('static/plots/ac_vs_consumption.png', dpi=150)
    plt.close()

    # 3. Outside Temperature vs Electricity Consumption
    plt.figure(figsize=(7, 4.5))
    sns.regplot(data=df, x='Outside_Temperature', y='Electricity_Consumption',
                scatter_kws={'alpha': 0.5, 'color': '#f97316', 's': 30},
                line_kws={'color': '#b91c1c', 'linewidth': 2})
    plt.title('Outside Temperature (°C) vs Electricity Consumption', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Outside Temperature (°C)', fontsize=11)
    plt.ylabel('Electricity Consumption (kWh/day)', fontsize=11)
    plt.tight_layout()
    plt.savefig('static/plots/temp_vs_consumption.png', dpi=150)
    plt.close()

    # 4. Previous Consumption vs Actual Consumption
    plt.figure(figsize=(7, 4.5))
    sns.regplot(data=df, x='Previous_Consumption', y='Electricity_Consumption',
                scatter_kws={'alpha': 0.5, 'color': '#8b5cf6', 's': 30},
                line_kws={'color': '#4c1d95', 'linewidth': 2})
    plt.title('Previous Consumption vs Current Electricity Consumption', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Previous Consumption (kWh/day)', fontsize=11)
    plt.ylabel('Current Electricity Consumption (kWh/day)', fontsize=11)
    plt.tight_layout()
    plt.savefig('static/plots/prev_vs_actual.png', dpi=150)
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True,
                linewidths=0.8, linecolor='white', square=True)
    plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('static/plots/correlation_heatmap.png', dpi=150)
    plt.close()

    print("All EDA visualizations saved successfully in static/plots/.")

def train_and_evaluate():
    """
    Execute complete ML Pipeline:
    1. Load dataset
    2. Preprocess & check data
    3. Train-test split (80:20)
    4. Train Linear Regression, Decision Tree, Random Forest
    5. Evaluate MAE, MSE, RMSE, R²
    6. Select and save best model
    7. Generate model comparison & Actual vs Predicted plots
    """
    csv_path = 'data/household_electricity.csv'
    if not os.path.exists(csv_path):
        df = generate_synthetic_dataset(csv_path, n_samples=600)
    else:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing dataset from {csv_path} ({len(df)} rows).")

    # Generate EDA plots
    generate_eda_plots(df)

    # Preprocessing Checks
    missing_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())
    total_samples = len(df)
    features = ['People', 'AC_Usage_Hours', 'Number_of_Fans', 'Number_of_Appliances', 
                'Daily_Usage_Hours', 'Outside_Temperature', 'Previous_Consumption']
    target = 'Electricity_Consumption'

    X = df[features]
    y = df[target]

    # 80:20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Models dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=6, min_samples_split=5, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)
    }

    results = {}
    test_predictions = {}

    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_predictions[name] = preds

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        results[name] = {
            'MAE': round(float(mae), 4),
            'MSE': round(float(mse), 4),
            'RMSE': round(float(rmse), 4),
            'R2': round(float(r2), 4)
        }
        print(f"[{name}] MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R² Score: {r2:.4f}")

    # Model Selection: Lowest RMSE (or highest R²)
    best_model_name = min(results.keys(), key=lambda k: results[k]['RMSE'])
    best_model = models[best_model_name]
    best_predictions = test_predictions[best_model_name]
    print(f"\n>> Best Model Selected: {best_model_name} (Lowest Test RMSE: {results[best_model_name]['RMSE']})")

    # Save the Best Model
    model_path = 'models/electricity_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"Saved best model ({best_model_name}) to {model_path}")

    # Generate Model Comparison Chart
    plt.figure(figsize=(8, 4.5))
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.25

    rmse_scores = [results[m]['RMSE'] for m in model_names]
    mae_scores = [results[m]['MAE'] for m in model_names]
    r2_scores = [results[m]['R2'] for m in model_names]

    plt.bar(x - width, rmse_scores, width, label='RMSE (kWh)', color='#ef4444')
    plt.bar(x, mae_scores, width, label='MAE (kWh)', color='#3b82f6')
    plt.bar(x + width, r2_scores, width, label='R² Score', color='#10b981')

    plt.xlabel('Regression Models', fontsize=11, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=11, fontweight='bold')
    plt.title('Model Performance Comparison (Test Set)', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, [m.replace(' Regressor', '') for m in model_names], fontsize=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('static/plots/model_comparison.png', dpi=150)
    plt.close()

    # Generate Actual vs Predicted Graph for Best Model
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_predictions, color='#2563eb', alpha=0.65, edgecolors='k', s=45, label='Predicted vs Actual')
    min_val = min(y_test.min(), best_predictions.min())
    max_val = max(y_test.max(), best_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    plt.title(f'Actual vs. Predicted Consumption ({best_model_name})', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Actual Electricity Consumption (kWh/day)', fontsize=11)
    plt.ylabel('Predicted Electricity Consumption (kWh/day)', fontsize=11)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig('static/plots/actual_vs_predicted.png', dpi=150)
    plt.close()

    # Save summary metadata for the Web Application
    metadata = {
        'dataset_info': {
            'total_samples': total_samples,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': features,
            'target': target,
            'missing_values': missing_counts,
            'duplicates': duplicates_count,
            'is_synthetic': True
        },
        'model_comparison': results,
        'best_model': {
            'name': best_model_name,
            'metrics': results[best_model_name]
        },
        'sample_records': df.head(5).to_dict(orient='records')
    }

    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Metadata and evaluation metrics saved to models/metadata.json")
    print("Training pipeline finished successfully!\n")

if __name__ == '__main__':
    train_and_evaluate()
