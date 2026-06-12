import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

# LOAD DATA
df = pd.read_csv("../train_with_aug_raw.csv")

# LABEL DISTRIBUTION
label_counts = df["label"].value_counts()

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar")

plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")

plt.show()