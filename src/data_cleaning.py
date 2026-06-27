import pandas as pd

data = pd.read_csv("dataset/diabetes.csv")

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns:
    print(f"\n{col} Zero Values:")
    print((data[col] == 0).sum())