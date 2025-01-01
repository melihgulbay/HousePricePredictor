import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold

def evaluate_model(model, X, y, X_scaled, needs_scaling):
    """
    Evaluates a model using various metrics.
    """
    # Convert to numpy arrays for consistent handling
    if needs_scaling:
        X_eval = X_scaled.to_numpy()
    else:
        X_eval = X.to_numpy()
    y_array = y.to_numpy()
    
    # Use cross-validation for more robust evaluation
    cv_scores = cross_val_score(model, X_eval, y_array, cv=5, scoring='r2')
    
    # Split data for additional metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X_eval, y_array, test_size=0.2, random_state=42
    )
    
    # Fit on training data
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate prediction intervals (if applicable)
    prediction_std = None
    if hasattr(model, 'estimators_'):
        if isinstance(model, RandomForestRegressor):
            # For Random Forest, use individual tree predictions
            predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
            prediction_std = np.std(predictions, axis=0).mean()
        elif isinstance(model, GradientBoostingRegressor):
            # For Gradient Boosting, use staged predictions
            staged_preds = np.array(list(model.staged_predict(X_test)))
            prediction_std = np.std(staged_preds, axis=0).mean()
    
    return {
        'cv_score_mean': cv_scores.mean(),
        'cv_score_std': cv_scores.std(),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'prediction_std': prediction_std
    }

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
    
    # Convert to numpy array to avoid feature names warning
    X_array = X.to_numpy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled['m² (Brüt)'] = scaler.fit_transform(X[['m² (Brüt)']])
    X_scaled_array = X_scaled.to_numpy()
    
    # Dictionary to store model metrics
    model_metrics = {}
    
    # Train and evaluate Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_scaled_array, y)
    model_metrics['linear'] = evaluate_model(lr_model, X, y, X_scaled, True)
    
    # Train and evaluate Random Forest
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, 
                                   min_samples_split=10, min_samples_leaf=4,
                                   random_state=42, n_jobs=-1)
    rf_model.fit(X_array, y)
    model_metrics['random_forest'] = evaluate_model(rf_model, X, y, X_scaled, False)
    
    # Train and evaluate SVR
    svr_model = SVR(kernel='rbf', C=1000.0, epsilon=0.1, gamma='scale')
    svr_model.fit(X_scaled_array, y)
    model_metrics['svr'] = evaluate_model(svr_model, X, y, X_scaled, True)
    
    # Train and evaluate Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                        max_depth=5, min_samples_split=5,
                                        min_samples_leaf=3, random_state=42)
    gb_model.fit(X_array, y)
    model_metrics['gradient_boosting'] = evaluate_model(gb_model, X, y, X_scaled, False)
    
    # Train and evaluate XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Manual cross-validation for XGBoost
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X_array):
        X_train_cv, X_val_cv = X_array[train_idx], X_array[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        xgb_model.fit(X_train_cv, y_train_cv)
        
        # Predict and calculate R2 score
        y_pred_cv = xgb_model.predict(X_val_cv)
        cv_scores.append(r2_score(y_val_cv, y_pred_cv))
    
    # Train final model on full training data
    X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.2, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Calculate predictions
    y_pred = xgb_model.predict(X_test)
    
    # Calculate metrics for XGBoost
    xgb_metrics = {
        'cv_score_mean': np.mean(cv_scores),
        'cv_score_std': np.std(cv_scores),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'prediction_std': None
    }
    model_metrics['xgboost'] = xgb_metrics
    
    # Save models and metrics
    with open('house_price_models.pkl', 'wb') as f:
        pickle.dump({
            'linear': (lr_model, X_scaled.columns, scaler),
            'random_forest': (rf_model, X.columns, None),
            'svr': (svr_model, X_scaled.columns, scaler),
            'gradient_boosting': (gb_model, X.columns, None),
            'xgboost': (xgb_model, X.columns, None),
            'metrics': model_metrics
        }, f)
    
    return {
        'linear': (lr_model, X_scaled.columns),
        'random_forest': (rf_model, X.columns),
        'svr': (svr_model, X_scaled.columns),
        'gradient_boosting': (gb_model, X.columns),
        'xgboost': (xgb_model, X.columns)
    }
