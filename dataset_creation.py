# dataset_creation.py
import pandas as pd
import numpy as np

def dataset_create():
    """Generates the synthetic medical cost data with multicollinearity."""
    np.random.seed(42)
    N = 1000

    Age = np.random.randint(18, 65, N)
    BMI = np.random.normal(loc=30, scale=5, size=N)
    # Multicollinearity: Obesity_Score is just a noisy version of BMI
    Obesity_Score = BMI * 1.5 + np.random.normal(0, 1, N)
    Smoker = np.random.randint(0, 2, N)
    Children = np.random.randint(0, 6, N)

    # Create the charges based on a linear model with noise
    charges = (
        5000  # Base cost (Intercept)
        + 200 * Age
        + 350 * BMI
        + 50 * Obesity_Score  # The correlated feature
        + 20000 * Smoker
        + 500 * Children
        + np.random.normal(0, 5000, N)  # Add significant noise
    )
    charges[charges < 0] = 0

    data = pd.DataFrame({
        'Age': Age,
        'BMI': BMI,
        'Obesity_Score': Obesity_Score,
        'Smoker': Smoker,
        'Children': Children,
        'Charges': charges
    })
    # Save the dataset to CSV
    data.to_csv('insurance.csv', index=False)
    print("Dataset 'insurance.csv' created.")
    return data

if __name__ == "__main__":
    dataset_create()
