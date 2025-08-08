from flask import Flask, render_template, request
from ga_logic import (
    load_food_data, GA_CONFIG, GeneticAlgorithm, AGAScenario1,
    calculate_mape, calculate_smape, calculate_r2
)
import numpy as np
from translations import TEXT

app = Flask(__name__)

food_df = load_food_data()

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def home():    
    lang = request.args.get('lang', 'en')
    if lang not in ['en', 'id']:
        lang = 'en'
    return render_template('index.html', text=TEXT[lang], lang=lang)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:        
        lang = request.form.get('lang', 'en')
        if lang not in ['en', 'id']:
            lang = 'en'

        user_profile = {
            'age': int(request.form['age']),
            'weight_kg': float(request.form['weight_kg']),
            'height_cm': float(request.form['height_cm']),
            'trimester': int(request.form['trimester']),
            'medical_conditions': {
                'hypertension': 'hypertension' in request.form,
                'diabetes': 'diabetes' in request.form,
                'anemia': 'anemia' in request.form,
                'pre_eclampsia': 'pre_eclampsia' in request.form,
                'inflammation': 'inflammation' in request.form
            }
        }
        algorithm_choice = request.form.get('algorithm', 'aga')
    except (KeyError, ValueError) as e:
        return f"Form Error: Missing or invalid data provided. Details: {e}", 400

    web_ga_config = GA_CONFIG.copy()
    web_ga_config['NUM_GENERATIONS'] = 50

    if algorithm_choice == 'ga':
        model = GeneticAlgorithm(food_df, web_ga_config)
    else:
        model = AGAScenario1(food_df, web_ga_config)

    best_plan, actual_nutrients, target_needs, best_fitness = model.run(user_profile)

    structured_plan = {}
    current_slot = 0
    for meal, slots in web_ga_config['MEAL_SLOTS'].items():
        structured_plan[meal] = best_plan[current_slot : current_slot + slots]
        current_slot += slots

    comparison_data = []
    y_true, y_pred = [], []
    for key, target_val in target_needs.items():
        actual_val = actual_nutrients.get(key, 0)
        y_true.append(target_val)
        y_pred.append(actual_val)
        comparison_data.append({
            'nutrient': key,
            'target': target_val,
            'actual': actual_val,
            'difference': actual_val - target_val
        })

    metrics = {
        'mape': calculate_mape(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred)
    }

    return render_template(
        'results.html',
        text=TEXT[lang],
        lang=lang,
        structured_plan=structured_plan,
        algorithm=algorithm_choice,
        fitness=best_fitness,
        user_profile=user_profile,
        comparison_data=comparison_data,
        metrics=metrics
    )

if __name__ == '__main__':
    app.run(debug=True)
