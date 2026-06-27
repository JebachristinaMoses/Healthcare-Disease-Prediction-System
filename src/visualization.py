from turtle import st

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/diabetes.csv")

data["Outcome"].value_counts().plot(kind="bar")

plt.title("Diabetes Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")

plt.show()

st.divider()

st.subheader("Feature Importance")

st.image(
    "screenshots/feature_importance.png",
    use_container_width=True
)