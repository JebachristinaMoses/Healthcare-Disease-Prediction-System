'''import pandas as pd

# Load dataset
data = pd.read_csv("dataset/diabetes.csv")

print("\nFirst 5 Records")
print(data.head())

print("\nDataset Shape")
print(data.shape)

print("\nColumns")
print(data.columns)

print("\nDataset Information")
print(data.info())

print("\nStatistical Summary")
print(data.describe())

print("\nOutcome Counts")
print(data["Outcome"].value_counts())

print("\nMissing Values")
print(data.isnull().sum())

print("\nDuplicate Records")
print(data.duplicated().sum())'''

import pandas as pd

data = pd.read_csv("dataset/diabetes.csv")

print("\nAverage Glucose Level:")
print(data["Glucose"].mean())

print("\nAverage BMI:")
print(data["BMI"].mean())

print("\nAverage Age:")
print(data["Age"].mean())

print("\nHighest Glucose Level:")
print(data["Glucose"].max())

print("\nLowest Glucose Level:")
print(data["Glucose"].min())

print("\nHighest BMI:")
print(data["BMI"].max())

print("\nLowest BMI:")
print(data["BMI"].min())