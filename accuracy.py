import pandas as pd
from sklearn.metrics import accuracy_score

df_ground_truth = pd.read_excel('test.xlsx')
df_predictions = pd.read_csv('test_results.csv')

df_merged = pd.merge(
    df_ground_truth,
    df_predictions,
    left_on='name',
    right_on='Filename'
)

accuracy = accuracy_score(df_merged['tag'], df_merged['Prediction'])

print(f"Total Samples: {len(df_merged)}")
print(f"Correct Predictions: {(df_merged['tag'] == df_merged['Prediction']).sum()}")
print(f"Accuracy: {accuracy * 100:.2f}%")
