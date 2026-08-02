import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    print("--- Starting Data Preparation ---")
    data_path = "tourism_project/data/tourism.csv"
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        print("Data loaded successfully.")
        
        # Data Cleaning: Remove unnecessary columns
        # CustomerID is a unique identifier and holds no predictive value
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
            print("Dropped 'CustomerID' column.")
            
        # Drop duplicates to prevent data leakage and bias
        initial_shape = df.shape
        df = df.drop_duplicates()
        print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
        
        # Separate features (X) and target variable (y)
        X = df.drop('ProdTaken', axis=1)
        y = df['ProdTaken']
        
        # Split the data into training and testing sets (80% train, 20% test)
        # Using stratify=y to maintain the proportion of classes in the target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Save the splits locally
        # These will be picked up as artifacts by the GitHub Actions workflow
        X_train.to_csv("Xtrain.csv", index=False)
        X_test.to_csv("Xtest.csv", index=False)
        y_train.to_csv("ytrain.csv", index=False)
        y_test.to_csv("ytest.csv", index=False)
        
        print("Data preparation completed. Train and test splits saved locally as CSV files.")
        
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    prepare_data()
