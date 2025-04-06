from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

# Flask App Initialization
app = Flask(__name__)

# Set base directory to the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the relevant CSV files using absolute paths
forecast_df = pd.read_csv(os.path.join(BASE_DIR, "data", "demand_forecasting.csv"))
pricing_df = pd.read_csv(os.path.join(BASE_DIR, "data", "pricing_optimization.csv"))
inventory_df = pd.read_csv(os.path.join(BASE_DIR, "data", "inventory_monitoring.csv"))
alerts_df = pd.read_csv(os.path.join(BASE_DIR, "data", "inventory_alerts.csv"))

# Load trained Random Forest model
with open(os.path.join(BASE_DIR, "models", "rf_model.pki"), "rb") as f:
    model = pickle.load(f)

# Forecasting logic
def forecast_demand():
    top_demand = forecast_df.sort_values(by='Sales Quantity', ascending=False)
    return top_demand[['Product ID', 'Store ID', 'Sales Quantity']]


# Pricing optimization logic
def optimal_price():
    # Add a new 'Profit' column (example formula: Sales Volume * (Price - Discounts) - Storage Cost)
    pricing_df['Profit'] = (
        (pricing_df['Sales Volume'] * (pricing_df['Price'] - pricing_df['Discounts']))
        - pricing_df['Storage Cost']
    )
    best_price = pricing_df.loc[pricing_df['Profit'].idxmax()]
    return best_price['Price'], best_price['Profit']


# Predict sales volume using the ML model
def predict_sales_volume(price, stock, promotion, rating, reviews):
    input_data = pd.DataFrame([{
        'Price': price,
        'Stock': stock,
        'Promotion': promotion,
        'Rating': rating,
        'Reviews': reviews
    }])
    prediction = model.predict(input_data)
    return prediction[0]

# Reorder plan logic for all inventory
def generate_reorder_plan():
    plan = []
    for _, row in inventory_df.iterrows():
        stock = row['Stock Levels']
        reorder_point = row['Reorder Point']
        predicted_demand = row.get('Predicted Demand', 100)
        reorder_qty = 0
        if stock < reorder_point:
            reorder_qty = max(predicted_demand - stock, 0)
        plan.append({
            'Product ID': row['Product ID'],
            'Stock Levels': stock,
            'Reorder Point': reorder_point,
            'Reorder Quantity': reorder_qty
        })
    return plan

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast')
def forecast():
    data = forecast_demand().to_dict(orient='records')
    return render_template('forecast.html', forecasts=data)

@app.route('/pricing')
def pricing():
    price, profit = optimal_price()
    return render_template('pricing.html', price=price, profit=profit)

@app.route('/predict', methods=['POST'])
def predict():
    price = float(request.form['price'])
    stock = int(request.form['stock'])
    promotion = int(request.form['promotion'])
    rating = float(request.form['rating'])
    reviews = int(request.form['reviews'])
    prediction = predict_sales_volume(price, stock, promotion, rating, reviews)
    return render_template('pricing.html', predicted_sales=prediction)

@app.route('/inventory')
def inventory():
    plan = generate_reorder_plan()
    return render_template('inventory.html', reorder_plan=plan, alerts=alerts_df.to_dict(orient='records'))

# Main Entry
if __name__ == '__main__':
    app.run(debug=True)
