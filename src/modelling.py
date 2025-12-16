from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

"""# **Data Splitting**"""

    test_path = 'res/test.csv'
    df_test = pd.read_csv(test_path)
def automate_model():


    X = df_train.drop(columns=target_column)
    y = df_train['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """# **Modelling**"""

    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)

    """## HyperParameter Tuning"""

    param_dist = {
        'n_estimators': np.linspace(100, 500, 5, dtype=int),
        'max_depth': np.linspace(10, 50, 5, dtype=int),
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    random_search = RandomizedSearchCV(estimator=model_rf, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_rf_rs = random_search.best_estimator_

    """# **Inference**"""


    X = df_test.drop(columns='PassengerId')
    y = df_test['PassengerId']

    X = preprocessor.fit_transform(X)
    feature_names = preprocessor.named_steps['transformer'].get_feature_names_out()

    df_test = pd.DataFrame(X, columns=feature_names)
    df_test['PassengerId'] = y.reset_index(drop=True)

    predictions = best_rf_rs.predict(df_test.drop(columns='PassengerId'))
    df_final = pd.DataFrame(df_test['PassengerId'])
    df_final['Survived'] = predictions

    df_final.to_csv(final_path, index=False)