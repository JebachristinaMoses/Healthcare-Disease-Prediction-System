import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
data = pd.read_csv("dataset/diabetes_cleaned.csv")

columns = ["Glucose", "BMI", "Age", "Insulin"]

for col in columns:
    plt.figure(figsize=(8,5))
    plt.hist(data[col], bins=20)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
data = pd.read_csv("dataset/diabetes_cleaned.csv")

# Create screenshots folder if it doesn't exist
os.makedirs("screenshots", exist_ok=True)

# Features for histogram
columns = ["Glucose", "BMI", "Age", "Insulin"]

for col in columns:
    plt.figure(figsize=(8, 5))

    plt.hist(data[col], bins=20)

    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    # Save image
    plt.savefig(f"screenshots/{col.lower()}_histogram.png", dpi=300)

    # Close plot
    plt.close()

print("All histograms saved successfully!")

