import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

if __name__ == "__main__":
    """# **Data Splitting**"""

    target_column = ['PassengerId', 'Survived']
    train_path = 'res/train_preprocess.csv'
    df_train = pd.read_csv(train_path)
    
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
        
        mlflow.sklearn.log_model(
            sk_model=model_rf,
            artifact_path=model_rf,
            input_example=input_example
        )