# OptiCrop: Smart Agricultural Production Optimization Engine

OptiCrop is an AI-powered agricultural recommendation and suitability assessment platform. By combining machine learning classifiers (Random Forest) with analytical Chart.js dashboards, the engine evaluates soil macronutrients and climate parameters to help farmers, researchers, and policymakers maximize yield efficiency.

---

## 🚀 Key Features

* **Scenario 1: Smart Crop Recommendation**
  * Enter Nitrogen (N), Phosphorus (P), Potassium (K), pH, temperature, humidity, and rainfall to receive a real-time optimal crop prediction.
  * Powered by a trained **Random Forest Classifier** achieving **99.32% accuracy**.

* **Scenario 2: Crop Suitability Assessor**
  * Select a target crop and input soil parameters to assess compatibility.
  * Outputs detailed metrics score gauges, parameter thresholds, and tailored agronomic suggestions.

* **Scenario 3: Research & Policy Dashboard**
  * Toggles comparative distribution plots.
  * Interactive Chart.js charts comparing climate margins, nutrient bounds, and pH requirements.

---

## 📂 Project Architecture

```
/OptiCrop-Smart-Agricultural-Production-Optimization-Engine-Smartbridge
|-- 1. Brainstorming & Ideation
|-- 2. Requirement Analysis
|-- 3. Project Design Phase
|-- 4. Project Planning Phase
|-- 5. Project Development Phase
|   |-- app.py (Flask Web Server)
|   |-- train_model.py (Model Training Pipeline)
|   |-- model.pkl (Serialized ML Model)
|   |-- crop_stats.json (Optimal Ranges Database)
|   |-- Crop_recommendation.csv (Agricultural Dataset)
|   |-- static/ (CSS Styles & Web Assets)
|   `-- templates/ (Jinja2 HTML Interfaces)
|-- 6.Project Testing
|-- 7.Project Documentation
`-- 8.Project Demonstration
```

---

## 💻 Local Setup & Execution

### Pre-requisites
* Python 3.10 or higher
* Modern web browser (Chrome, Edge, Safari, Firefox)

### Step 1: Open the Terminal
Navigate to the project development directory:
```bash
cd "5. Project Development Phase"
```

### Step 2: Activate the Virtual Environment
Activate the pre-configured Python virtual environment:
```bash
source venv/bin/activate
```

### Step 3: Run the Application
Start the local Flask development server:
```bash
python app.py
```
The server will boot on: **`http://127.0.0.1:5001`**

---

## ⚡ Performance Testing (Locust)
* **Concurrent swarmed users**: 50 virtual users at 5 users/sec spawn rate.
* **Average Latency**: **9.33 milliseconds** (under 10 ms!).
* **Maximum Latency**: **52 milliseconds**.
* **Request Failures**: **0%** (0 failed requests out of 512+ runs).
