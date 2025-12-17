import pandas as pd
from joblib import load

test_path = 'res/test.csv'
df_test = pd.read_csv(test_path)

pipeline_path = 'res/preprocessor_pipeline.joblib'
preprocessor = load(pipeline_path)

model_path = 'res/model.h5'
best_rf_rs = load()

final_path = 'res/submission.csv'

"""# **Inference**"""

X = df_test.drop(columns='PassengerId')
y = df_test['PassengerId']

X = preprocessor.transform(X)
feature_names = preprocessor.named_steps['transformer'].get_feature_names_out()

df_test = pd.DataFrame(X, columns=feature_names)
df_test['PassengerId'] = y.reset_index(drop=True)

predictions = best_rf_rs.predict(df_test.drop(columns='PassengerId'))
df_final = pd.DataFrame(df_test['PassengerId'])
df_final['Survived'] = predictions

df_final.to_csv(final_path, index=False)