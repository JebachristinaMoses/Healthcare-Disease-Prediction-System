import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
data = pd.read_csv("dataset/diabetes_cleaned.csv")

# Create screenshots folder if not exists
os.makedirs("screenshots", exist_ok=True)

# Correlation matrix
correlation = data.corr()

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

# Save image
plt.savefig(
    "screenshots/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Correlation heatmap saved successfully!")