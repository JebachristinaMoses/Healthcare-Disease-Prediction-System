import pandas as pd

data = pd.read_csv("dataset/diabetes_cleaned.csv")

correlation = data.corr()

print(correlation["Outcome"].sort_values(ascending=False))