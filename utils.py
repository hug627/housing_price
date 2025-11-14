import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Preparing data for regression models
def prepare_data(housing):
    """Split data into train and test sets"""
    X = housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
                 'Avg. Area Number of Bedrooms', 'Area Population']]
    Y = housing['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Training Linear Regression model
def train_linear_regression(housing):
    """Train and evaluate Linear Regression model"""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(housing)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'model': model,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
    }
    return results


# Training Lasso Regression model
def train_lasso_regression(housing):
    """Train and evaluate Lasso Regression model"""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(housing)
    
    # Train Lasso model
    lasso_model = LassoCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    lasso_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = lasso_model.predict(X_train)
    y_test_pred = lasso_model.predict(X_test)
    
    # Calculate training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'model': lasso_model,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
    }
    return results


# Training Ridge Regression model
def train_ridge_regression(housing):
    """Train and evaluate Ridge Regression model"""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(housing)
    
    # Train Ridge model
    ridge_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge_model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = ridge_model.predict(X_train)
    y_test_pred = ridge_model.predict(X_test)
    
    # Calculate training metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate test metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results = {
        'model': ridge_model,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
        }
    }
    return results


# Example usage:
# linear_results = train_linear_regression(housing)
# ridge_results = train_ridge_regression(housing)
# lasso_results = train_lasso_regression(housing)
#
# print("Linear Regression R2:", linear_results['metrics']['test']['r2'])
# print("Ridge Regression R2:", ridge_results['metrics']['test']['r2'])
# print("Lasso Regression R2:", lasso_results['metrics']['test']['r2'])