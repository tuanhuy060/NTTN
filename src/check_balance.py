import pandas as pd

df = pd.read_csv("../train_with_aug_raw.csv")

print(df["label"].value_counts())

print()

print(
    df["label"].value_counts(normalize=True)*100
)