import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Dummy data for training
df = pd.DataFrame({
    'Price': [10, 20, 30, 40, 50],
    'Stock': [100, 80, 60, 40, 20],
    'Promotion': [0, 1, 0, 1, 1],
    'Rating': [4.5, 4.0, 3.5, 4.2, 3.8],
    'Reviews': [200, 150, 100, 50, 300],
    'SalesVolume': [500, 400, 300, 200, 100]
})

X = df[['Price', 'Stock', 'Promotion', 'Rating', 'Reviews']]
y = df['SalesVolume']

model = RandomForestRegressor()
model.fit(X, y)

# Save the model
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, 'rf_model.pki'), 'wb') as f:
    pickle.dump(model, f)
