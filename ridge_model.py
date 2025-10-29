# ridge_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump, load
from dataset_creation import dataset_create # Import to ensure data exists

def train_and_save_ridge(csv_file, test_size=0.2, random_state=42):
    """
    Loads data, trains the optimal Ridge Regression model using GridSearchCV,
    and saves the model. Assumes scaler and feature names are already saved by model.py.
    """
    print("\nStarting Ridge Regression Model Training...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Running dataset_create().")
        df = dataset_create()
    
    X = df.drop('Charges', axis=1)
    y = df['Charges']

    # 2. Split Data and Load Scaler (assuming model.py handles initial scaling setup)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    try:
        # Load the scaler fitted on the OLS training data
        scaler = load('standard_scaler.joblib')
    except FileNotFoundError:
        # If the scaler wasn't saved, fit and transform it here
        print("Scaler not found. Fitting scaler now (This should be handled by model.py in a clean workflow).")
        scaler = StandardScaler()
        scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Setup GridSearchCV for Ridge
    ridge = Ridge(random_state=random_state)
    # Search for the optimal alpha (lambda)
    parameters = {'alpha': np.logspace(-4, 2, 20)} 
    
    # Use 5-fold cross-validation to find the best alpha, optimizing for minimum MSE
    ridge_regressor = GridSearchCV(
        ridge, parameters, 
        scoring='neg_mean_squared_error', # Maximize negative MSE (minimize positive MSE)
        cv=5
    )

    # 4. Train the Optimal Ridge Model
    ridge_regressor.fit(X_train_scaled, y_train)
    
    best_ridge_model = ridge_regressor.best_estimator_

    # 5. Predict and Evaluate
    y_pred_ridge = best_ridge_model.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test, y_pred_ridge)
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)

    print(f"\n--- Ridge Evaluation ---")
    print(f"Optimal alpha found: {ridge_regressor.best_params_['alpha']:.4f}")
    print(f"Test R-squared: {ridge_r2:.4f}")
    print(f"Test Mean Squared Error: {ridge_mse:,.2f}")

    # 6. Save the trained Ridge model
    dump(best_ridge_model, 'ridge_model.joblib')
    print("\nOptimal Ridge Model saved as 'ridge_model.joblib'")


if __name__ == "__main__":
    train_and_save_ridge("insurance.csv")
