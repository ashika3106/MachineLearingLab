import os
import time
import subprocess
import requests
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = "report_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

PORT = 5050
BASE_URL = f"http://127.0.0.1:{PORT}"

def wait_for_server(url=BASE_URL, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print(f"Flask server is up and responsive at {url}!")
                return True
        except Exception:
            time.sleep(0.5)
    return False

def generate_vscode_explorer_image(output_path):
    """
    Generate an authentic, high-resolution dark-themed VS Code explorer image.
    """
    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {
        margin: 0;
        padding: 0;
        background-color: #181818;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #cccccc;
        font-size: 13px;
        user-select: none;
        width: 320px;
    }
    .sidebar {
        background-color: #181818;
        border-right: 1px solid #2b2b2b;
        padding: 8px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.5);
    }
    .sidebar-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #bbbbbb;
        padding: 6px 16px 8px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .project-header {
        font-size: 11px;
        font-weight: 700;
        color: #ffffff;
        padding: 4px 12px;
        display: flex;
        align-items: center;
        gap: 6px;
        background-color: #202020;
    }
    .tree {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .tree-item {
        display: flex;
        align-items: center;
        padding: 3px 12px 3px 20px;
        gap: 6px;
        font-size: 13px;
        cursor: pointer;
    }
    .tree-item.folder {
        font-weight: 600;
        color: #e0e0e0;
    }
    .tree-item.nested {
        padding-left: 36px;
    }
    .tree-item.deep-nested {
        padding-left: 50px;
    }
    .tree-item:hover, .tree-item.active {
        background-color: #2a2d2e;
        color: #ffffff;
    }
    .icon {
        font-size: 12px;
        display: inline-block;
        width: 16px;
        text-align: center;
    }
    .chevron {
        font-size: 9px;
        color: #aaaaaa;
        display: inline-block;
        width: 10px;
    }
    .py-icon { color: #3572A5; }
    .csv-icon { color: #2ecc71; }
    .json-icon { color: #f1c40f; }
    .html-icon { color: #e67e22; }
    .css-icon { color: #5dade2; }
    .db-icon { color: #9b59b6; }
    .md-icon { color: #42b983; }
    .txt-icon { color: #95a5a6; }
    .pkl-icon { color: #e74c3c; }
    .img-icon { color: #1abc9c; }
</style>
</head>
<body>
<div class="sidebar">
    <div class="sidebar-header">
        <span>Explorer</span>
        <span style="letter-spacing: normal; color: #888;">•••</span>
    </div>
    <div class="project-header">
        <span class="chevron">▼</span>
        <span>HOUSEHOLD_ELECTRICITY_PREDICTION</span>
    </div>
    <ul class="tree">
        <li class="tree-item folder">
            <span class="chevron">▼</span>
            <span style="color:#dcb67a;">📁</span>
            <span>data</span>
        </li>
        <li class="tree-item nested">
            <span class="icon csv-icon">📊</span>
            <span>household_electricity.csv</span>
        </li>
        <li class="tree-item folder">
            <span class="chevron">▼</span>
            <span style="color:#dcb67a;">📁</span>
            <span>models</span>
        </li>
        <li class="tree-item nested">
            <span class="icon pkl-icon">🧠</span>
            <span>electricity_model.pkl</span>
        </li>
        <li class="tree-item nested">
            <span class="icon json-icon">{ }</span>
            <span>metadata.json</span>
        </li>
        <li class="tree-item folder">
            <span class="chevron">▼</span>
            <span style="color:#dcb67a;">📁</span>
            <span>static</span>
        </li>
        <li class="tree-item folder nested">
            <span class="chevron">▼</span>
            <span style="color:#dcb67a;">📁</span>
            <span>plots</span>
        </li>
        <li class="tree-item deep-nested">
            <span class="icon img-icon">📈</span>
            <span>actual_vs_predicted.png</span>
        </li>
        <li class="tree-item deep-nested">
            <span class="icon img-icon">📈</span>
            <span>ac_vs_consumption.png</span>
        </li>
        <li class="tree-item deep-nested">
            <span class="icon img-icon">📈</span>
            <span>consumption_dist.png</span>
        </li>
        <li class="tree-item deep-nested">
            <span class="icon img-icon">📈</span>
            <span>correlation_heatmap.png</span>
        </li>
        <li class="tree-item deep-nested">
            <span class="icon img-icon">📈</span>
            <span>model_comparison.png</span>
        </li>
        <li class="tree-item deep-nested">
            <span class="icon img-icon">📈</span>
            <span>temp_vs_consumption.png</span>
        </li>
        <li class="tree-item nested">
            <span class="icon css-icon">#</span>
            <span>style.css</span>
        </li>
        <li class="tree-item folder">
            <span class="chevron">▼</span>
            <span style="color:#dcb67a;">📁</span>
            <span>templates</span>
        </li>
        <li class="tree-item nested">
            <span class="icon html-icon">&lt;&gt;</span>
            <span>index.html</span>
        </li>
        <li class="tree-item active">
            <span class="icon py-icon">🐍</span>
            <span>app.py</span>
        </li>
        <li class="tree-item">
            <span class="icon py-icon">🐍</span>
            <span>train_model.py</span>
        </li>
        <li class="tree-item">
            <span class="icon db-icon">🗄️</span>
            <span>predictions.db</span>
        </li>
        <li class="tree-item">
            <span class="icon md-icon">📝</span>
            <span>README.md</span>
        </li>
        <li class="tree-item">
            <span class="icon txt-icon">📄</span>
            <span>requirements.txt</span>
        </li>
        <li class="tree-item">
            <span class="icon py-icon">🐍</span>
            <span>test_app.py</span>
        </li>
    </ul>
</div>
</body>
</html>
"""
    tmp_path = os.path.join(SCREENSHOT_DIR, "vscode_temp.html")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 340, "height": 650}, device_scale_factor=2)
        page.goto("file://" + os.path.abspath(tmp_path))
        page.wait_for_timeout(300)
        page.locator(".sidebar").screenshot(path=output_path)
        browser.close()
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("Generated VS Code directory structure image.")

def capture_all_screenshots():
    # 0. Generate VS Code directory structure screenshot
    vscode_path = os.path.join(SCREENSHOT_DIR, "fig1_vscode_structure.png")
    generate_vscode_explorer_image(vscode_path)

    # 1. Start Flask server on port 5050
    env = os.environ.copy()
    env["PORT"] = str(PORT)
    server_process = subprocess.Popen(["python", "app.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        if not wait_for_server():
            print("Failed to connect to Flask server.")
            return

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                device_scale_factor=2
            )
            page = context.new_page()

            # --- Webpage Home ---
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_timeout(800)

            # Fig 2: Dataset Overview & Sample Records Table
            fig1_elem = page.locator("#fig1-overview")
            fig1_elem.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig2_dataset_overview.png"))
            print("Captured: fig2_dataset_overview.png")

            # Fig 3: Data Preprocessing
            fig2_elem = page.locator("#preprocessing")
            fig2_elem.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig3_preprocessing.png"))
            print("Captured: fig3_preprocessing.png")

            # Fig 4: Exploratory Data Analysis (EDA)
            fig3_elem = page.locator("#eda")
            fig3_elem.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig4_eda.png"))
            print("Captured: fig4_eda.png")

            # Fig 5: Model Training & Comparison
            fig4_elem = page.locator("#models")
            fig4_elem.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig5_model_comparison.png"))
            print("Captured: fig5_model_comparison.png")

            # Fig 6: Selected Model Evaluation
            fig5_elem = page.locator("#evaluation")
            fig5_elem.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig6_evaluation.png"))
            print("Captured: fig6_evaluation.png")

            # Fig 7: Prediction Input Form (Default State)
            predict_section = page.locator(".prediction-section")
            predict_section.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig7_predict_form.png"))
            print("Captured: fig7_predict_form.png")

            # --- Submit Test Case 1 (Standard Household: 4 People, 5h AC, 3 Fans, 8 Appliances, 34°C, 10 kWh) ---
            page.fill("#people", "4")
            page.fill("#ac_usage_hours", "5.0")
            page.fill("#number_of_fans", "3")
            page.fill("#number_of_appliances", "8")
            page.fill("#daily_usage_hours", "10.0")
            page.fill("#outside_temperature", "34.0")
            page.fill("#previous_consumption", "10.0")
            page.wait_for_timeout(300)

            page.click("button[type='submit']")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(800)

            # Fig 8: Prediction Result (Test Case 1)
            page.locator(".prediction-section").screenshot(path=os.path.join(SCREENSHOT_DIR, "fig8_prediction_result_sample1.png"))
            print("Captured: fig8_prediction_result_sample1.png")

            # --- Submit Test Case 2 (High Summer Demand: 6 People, 10h AC, 5 Fans, 12 Appliances, 41.5°C, 30 kWh) ---
            page.fill("#people", "6")
            page.fill("#ac_usage_hours", "10.0")
            page.fill("#number_of_fans", "5")
            page.fill("#number_of_appliances", "12")
            page.fill("#daily_usage_hours", "16.0")
            page.fill("#outside_temperature", "41.5")
            page.fill("#previous_consumption", "30.0")
            page.wait_for_timeout(300)

            page.click("button[type='submit']")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(800)

            # Fig 9: Prediction Result (Test Case 2)
            page.locator(".prediction-section").screenshot(path=os.path.join(SCREENSHOT_DIR, "fig9_prediction_result_sample2.png"))
            print("Captured: fig9_prediction_result_sample2.png")

            # Fig 10: Prediction History Log (SQLite)
            history_elem = page.locator("#history")
            history_elem.screenshot(path=os.path.join(SCREENSHOT_DIR, "fig10_history_log.png"))
            print("Captured: fig10_history_log.png")

            browser.close()
            print("\n>> All screenshots captured with pixel perfection!")

    finally:
        server_process.terminate()
        server_process.wait()
        print("Flask server stopped.")

if __name__ == "__main__":
    capture_all_screenshots()
