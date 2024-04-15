# Assignment 1: Load ./cifar-10/class.csv
import pandas as pd

# Load csv
fn = "cifar-10/class.csv"
df = pd.read_csv(fn)
# print(df)
# print(df.head(), "\n") #これデータを参照しやすい
#print(df.shape)
#print(df.columns)
#print(df.dtypes)
labels = df["label"]
cls_names = df["class"]

# Print
print("labels:")
print(labels, "\n")
print("cls_names:")
print(cls_names)
