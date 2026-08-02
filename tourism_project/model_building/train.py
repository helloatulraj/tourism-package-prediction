import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import joblib
import mlflow
import os
import sys

def train_model():
    print("--- Starting Model Training with MLflow Tracking ---")
    
    try:
        # Load the train and test data
        X_train = pd.read_csv("Xtrain.csv")
        X_test = pd.read_csv("Xtest.csv")
        y_train = pd.read_csv("ytrain.csv").values.ravel()
        y_test = pd.read_csv("ytest.csv").values.ravel()
        
        # --- THE FIX: Drop any 'Unnamed' phantom columns ---
        X_train = X_train.loc[:, ~X_train.columns.str.contains('^Unnamed')]
        X_test = X_test.loc[:, ~X_test.columns.str.contains('^Unnamed')]
        
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessor = make_column_transformer(
            (StandardScaler(), numerical_cols),
            (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        )
        
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        pipeline = make_pipeline(preprocessor, model)
        
        param_grid = {
            'xgbclassifier__n_estimators': [50, 100, 150],
            'xgbclassifier__max_depth': [3, 5, 7],
            'xgbclassifier__learning_rate': [0.01, 0.1, 0.2]
        }
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Tourism_Package_Prediction_Exp")
        
        with mlflow.start_run():
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_pipeline = grid_search.best_estimator_
            
            y_pred = best_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("test_accuracy", accuracy)
            
            model_path = "tourism_project/deployment/best_model.joblib"
            joblib.dump(best_pipeline, model_path)
            print(f"Best model successfully saved to {model_path}")
            
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_model()
