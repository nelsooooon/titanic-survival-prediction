test_path = 'res/test.csv'
df_test = pd.read_csv(test_path)

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
        