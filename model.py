# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

def train_and_save_ols(csv_file, test_size=0.2, random_state=42):
    """
    Loads data, performs initial data preparation (scaling/splitting),
    trains the OLS Linear Regression model, and saves the model and scaler.
    """
    print("Starting OLS Linear Regression Model Training...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Ensure dataset_creation.py has been run.")
        return
    
    # Identify features (X) and target (y)
    X = df.drop('Charges', axis=1)
    y = df['Charges']
    feature_names = list(X.columns)

    # 2. Split Data (before scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Standardize Features and Save Scaler (CRITICAL FOR RIDGE/OLS COMPARISON)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler object for use in the Streamlit app
    dump(scaler, 'standard_scaler.joblib')
    print("StandardScaler saved as 'standard_scaler.joblib'")

    # Save feature names for Streamlit app
    dump(feature_names, 'feature_names.joblib')
    print("Feature names saved as 'feature_names.joblib'")

    # 4. Train OLS Model
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)

    # 5. Predict and Evaluate
    y_pred_ols = ols_model.predict(X_test_scaled)
    ols_r2 = r2_score(y_test, y_pred_ols)
    ols_mse = mean_squared_error(y_test, y_pred_ols)

    print(f"\n--- OLS Evaluation ---")
    print(f"Test R-squared: {ols_r2:.4f}")
    print(f"Test Mean Squared Error: {ols_mse:,.2f}")
    
    # Display coefficients to show potential instability due to multicollinearity
    ols_coeffs = pd.Series(ols_model.coef_, index=feature_names)
    print("\nOLS Coefficients:")
    print(ols_coeffs)
    
    # 6. Save the trained OLS model
    dump(ols_model, 'ols_model.joblib')
    print("\nOLS Model saved as 'ols_model.joblib'")


if __name__ == "__main__":
    # Ensure dataset is created before running models
    # This assumes dataset_creation.py is accessible or insurance.csv exists.
    # For a robust workflow, you would import and call dataset_create() here.
    train_and_save_ols("insurance.csv")
