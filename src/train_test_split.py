import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
data = pd.read_csv("dataset/diabetes_cleaned.csv")

# Features (Independent Variables)
X = data.drop("Outcome", axis=1)

# Target Variable (Dependent Variable)
y = data["Outcome"]

print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successfully!")

print(f"\nTraining Features Shape : {X_train.shape}")
print(f"Testing Features Shape  : {X_test.shape}")

print(f"\nTraining Labels Shape   : {y_train.shape}")
print(f"Testing Labels Shape    : {y_test.shape}")