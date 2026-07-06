from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# Load trained model and statistics
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
stats_path = os.path.join(os.path.dirname(__file__), 'crop_stats.json')

model = None
crop_stats = {}

if os.path.exists(model_path):
    model = joblib.load(model_path)
if os.path.exists(stats_path):
    with open(stats_path, 'r') as f:
        crop_stats = json.load(f)

# Helper function to get crop list
def get_crops():
    return sorted(list(crop_stats.keys())) if crop_stats else []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    prediction_text = None
    prediction_crop = None
    values = {}
    
    if request.method == 'POST':
        try:
            # Parse inputs
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            values = {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature, 'humidity': humidity,
                'ph': ph, 'rainfall': rainfall
            }

            # Validations
            if any(val < 0 for val in [N, P, K, temperature, humidity, ph, rainfall]):
                prediction_text = "Error: Input values cannot be negative."
            elif ph > 14:
                prediction_text = "Error: pH value must be between 0 and 14."
            elif humidity > 100:
                prediction_text = "Error: Humidity cannot exceed 100%."
            elif N == 0 and P == 0 and K == 0:
                prediction_text = "Error: Nutrient values (N, P, K) cannot all be zero."
            else:
                if model:
                    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                    prediction = model.predict(data)
                    prediction_crop = prediction[0]
                    prediction_text = f"Recommended Crop: {prediction_crop.capitalize()}"
                else:
                    prediction_text = "Model has not been trained yet. Please run train_model.py."
        except ValueError:
            prediction_text = "Error: Please enter valid numeric values for all parameters."

    return render_template('recommend.html', prediction_text=prediction_text, prediction_crop=prediction_crop, values=values)

@app.route('/suitability', methods=['GET', 'POST'])
def suitability():
    crops = get_crops()
    result = None
    values = {}
    selected_crop = None
    
    if request.method == 'POST':
        try:
            selected_crop = request.form['crop']
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            values = {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature, 'humidity': humidity,
                'ph': ph, 'rainfall': rainfall
            }

            if selected_crop not in crop_stats:
                result = {"error": f"Crop data for '{selected_crop}' not found."}
            elif any(val < 0 for val in [N, P, K, temperature, humidity, ph, rainfall]):
                result = {"error": "Input values cannot be negative."}
            elif ph > 14:
                result = {"error": "pH value must be between 0 and 14."}
            elif humidity > 100:
                result = {"error": "Humidity cannot exceed 100%."}
            else:
                # Compute suitability details
                crop_limits = crop_stats[selected_crop]
                suitability_details = {}
                score_sum = 0
                total_features = len(crop_limits)
                
                suggestions = []

                for feat, limits in crop_limits.items():
                    val = values[feat]
                    f_min, f_max, f_mean = limits['min'], limits['max'], limits['mean']
                    
                    # Compute a normalized score for this feature (1.0 if inside [min, max], decays otherwise)
                    if f_min <= val <= f_max:
                        feat_score = 100.0
                        status = "Optimal"
                    else:
                        # Calculate distance from closest bound
                        dist = min(abs(val - f_min), abs(val - f_max))
                        span = f_max - f_min if f_max != f_min else 1.0
                        feat_score = max(0.0, 100.0 - (dist / span) * 100.0)
                        
                        if val < f_min:
                            status = "Too Low"
                            suggestions.append(f"Increase {feat} (Current: {val}, Ideal Min: {f_min:.1f})")
                        else:
                            status = "Too High"
                            suggestions.append(f"Decrease {feat} (Current: {val}, Ideal Max: {f_max:.1f})")
                            
                    score_sum += feat_score
                    suitability_details[feat] = {
                        'status': status,
                        'score': round(feat_score, 1),
                        'min': round(f_min, 1),
                        'max': round(f_max, 1),
                        'mean': round(f_mean, 1),
                        'val': val
                    }
                
                overall_score = round(score_sum / total_features, 1)
                
                # Determine Suitability Grade
                if overall_score >= 85:
                    grade = "Excellent Compatibility"
                    grade_color = "success"
                elif overall_score >= 60:
                    grade = "Moderate Compatibility"
                    grade_color = "warning"
                else:
                    grade = "Poor Compatibility"
                    grade_color = "danger"

                if not suggestions:
                    suggestions.append(f"The soil and climatic conditions are ideal for cultivating {selected_crop}.")

                result = {
                    'overall_score': overall_score,
                    'grade': grade,
                    'grade_color': grade_color,
                    'details': suitability_details,
                    'suggestions': suggestions
                }
        except ValueError:
            result = {"error": "Please enter valid numeric values."}

    return render_template('suitability.html', crops=crops, result=result, values=values, selected_crop=selected_crop)

@app.route('/research')
def research():
    crops = get_crops()
    return render_template('research.html', crops=crops)

@app.route('/api/stats')
def api_stats():
    return jsonify(crop_stats)

if __name__ == '__main__':
    # Try reloading model and stats dynamically if they weren't loaded at startup
    if not model and os.path.exists(model_path):
        model = joblib.load(model_path)
    if not crop_stats and os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            crop_stats = json.load(f)
    app.run(debug=True, port=5002)
