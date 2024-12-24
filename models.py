import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

def train_models():
    # Read the CSV file with semicolon separator
    df = pd.read_csv('house_prices.csv', sep=';', encoding='utf-8')
    
    # Clean price column (remove "TL" and convert to float)
    df['Fiyat'] = df['Fiyat'].str.replace(' TL', '').str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Extract features (X) and target (y)
    X = df[['m² (Brüt)', 'Oda Sayısı']]
    
    # Convert both Bölge and Oda Sayısı to numeric using one-hot encoding
    X = pd.concat([
        X[['m² (Brüt)']], 
        pd.get_dummies(df['Oda Sayısı'], prefix='oda'),
        pd.get_dummies(df['Bölge'], prefix='bolge')
    ], axis=1)
    y = df['Fiyat']
    
    # Drop rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled['m² (Brüt)'] = scaler.fit_transform(X[['m² (Brüt)']])
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_scaled, y)
    
    # Train Random Forest model with better parameters
    rf_model = RandomForestRegressor(
        n_estimators=500,      # More trees for better averaging
        max_depth=10,          # Prevent overfitting
        min_samples_split=10,  # Require more samples per split
        min_samples_leaf=4,    # Require more samples per leaf
        random_state=42,
        n_jobs=-1             # Use all CPU cores
    )
    rf_model.fit(X, y)
    
    # Save both models
    with open('house_price_models.pkl', 'wb') as f:
        pickle.dump({
            'linear': (lr_model, X_scaled.columns, scaler),
            'random_forest': (rf_model, X.columns, None)
        }, f)
    
    return {'linear': (lr_model, X_scaled.columns), 'random_forest': (rf_model, X.columns)}
