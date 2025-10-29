# run_pipeline.py
from dataset_creation import dataset_create
from model import train_and_save_ols
from ridge_model import train_and_save_ridge

if __name__ == "__main__":
    # 1. Create the dataset
    print("--- STEP 1: Dataset Creation ---")
    dataset_create()
    
    # 2. Train and save OLS Model, Scaler, and Feature Names
    print("\n--- STEP 2: OLS Model Training & Saving Scaler ---")
    train_and_save_ols("insurance.csv")
    
    # 3. Train and save Optimal Ridge Model (uses saved scaler)
    print("\n--- STEP 3: Ridge Model Training & Saving Model ---")
    train_and_save_ridge("insurance.csv")
    
    print("\n\n--- PIPELINE COMPLETE ---")
    print("All necessary files (.csv and .joblib) are ready for the Streamlit app.")
