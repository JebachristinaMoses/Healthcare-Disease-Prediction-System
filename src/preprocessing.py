import pandas as pd

# Load dataset
data = pd.read_csv("dataset/diabetes.csv")

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zeros with median
for col in columns:
    median = data[col].replace(0, pd.NA).median()
    data[col] = data[col].replace(0, median)

print("Zero values replaced successfully!")

# Check again
for col in columns:
    print(f"{col}: {(data[col] == 0).sum()}")

# Save cleaned dataset
data.to_csv("dataset/diabetes_cleaned.csv", index=False)

print("\nCleaned dataset saved!")