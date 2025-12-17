import mlflow
import dagshub
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from joblib import dump
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # dagshub.init(repo_owner="nelsooooon", repo_name="titanic-survival-prediction", mlflow=True)
    # mlflow.set_tracking_uri("https://dagshub.com/nelsooooon/titanic-survival-prediction.mlflow")
    # mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Logging Model")    
    
    """# **Data Splitting**"""
    model_path = 'res/model.joblib'
    train_path = 'res/train_preprocess.csv'
    df_train = pd.read_csv(train_path)
    
    target_column = ['PassengerId', 'Survived']
    X = df_train.drop(columns=target_column)
    y = df_train['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_dist = {
        'n_estimators': np.linspace(100, 500, 5, dtype=int),
        'max_depth': np.linspace(10, 50, 5, dtype=int),
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    input_example = X_test[0:5]

    """# **Modelling**"""
    with mlflow.start_run(run_name="Random Search Forest"):
        mlflow.autolog()
        
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)

        """## HyperParameter Tuning"""

        random_search = RandomizedSearchCV(estimator=model_rf, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(X_train, y_train)

        best_rf_rs = random_search.best_estimator_
        y_pred = best_rf_rs.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        mlflow.sklearn.log_model(
            sk_model=best_rf_rs,
            name="modelRF",
            input_example=input_example
        )
        
        dump(best_rf_rs, model_path)