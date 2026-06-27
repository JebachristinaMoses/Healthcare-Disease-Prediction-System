import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv("dataset/diabetes_cleaned.csv")

os.makedirs("screenshots", exist_ok=True)

data["Outcome"].value_counts().plot(kind="bar")

plt.title("Diabetes Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")

plt.savefig("screenshots/diabetes_distribution.png", dpi=300)

plt.close()

print("Diabetes distribution chart saved!")