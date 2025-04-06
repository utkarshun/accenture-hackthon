import pandas as pd
import pickle

# Load the relevant CSV files
forecast_df = pd.read_csv("data/demand_forecasting.csv")
pricing_df = pd.read_csv("data/pricing_optimization.csv")
inventory_df = pd.read_csv("data/inventory_monitoring.csv")
alerts_df = pd.read_csv("data/inventory_alerts.csv")

# Load trained Random Forest model
with open("models/rf_model.pki", "rb") as f:
    model = pickle.load(f)

# Forecasting logic
def forecast_demand():
    top_demand = forecast_df.sort_values(by='Predicted Demand', ascending=False)
    return top_demand[['Product ID', 'Store ID', 'Predicted Demand']]

# Pricing optimization logic
def optimal_price():
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
        predicted_demand = row.get('Predicted Demand', 100)  # fallback if not present
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
