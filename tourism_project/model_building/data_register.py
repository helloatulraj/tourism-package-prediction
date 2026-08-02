import pandas as pd
import sys

def register_data():
    print("--- Starting Data Registration ---")
    data_path = "tourism_project/data/tourism.csv"
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        
        # Define the exact expected columns based on the Data Dictionary
        expected_columns = [
            'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 
            'Occupation', 'Gender', 'NumberOfPersonVisiting', 
            'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 
            'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 
            'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 
            'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch'
        ]
        
        # Validate columns: Check if any expected columns are missing
        missing_cols = [col for col in expected_columns if col not in df.columns]
        
        if missing_cols:
            print(f"Validation Failed! Missing columns: {missing_cols}")
            # Exit with status code 1 to intentionally fail the GitHub Actions pipeline
            sys.exit(1) 
        else:
            print("Validation Passed: All expected columns are present.")
            
        # Print a short summary of the data
        print("\n--- Dataset Summary ---")
        print(f"Total Rows: {df.shape[0]}")
        print(f"Total Columns: {df.shape[1]}")
        print("\nFirst 3 rows of the dataset:")
        print(df.head(3))
        
        print("\nData Registration completed successfully.")
        
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found. Please check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data registration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    register_data()
