import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
conn = sqlite3.connect("clicks.db")
df = pd.read_sql("SELECT * FROM clicks", conn)

print("\n================ DATA PREVIEW ================")
print(df.head())


print("\n================ BASIC STATISTICS ================")
print(df[["rating", "popularity"]].describe())


print("\n================ DISTRIBUTIONS ================")
print("Rating distribution:")
print(df["rating"].value_counts().sort_index())

print("\nPopularity distribution:")
print(df["popularity"].value_counts().sort_values(ascending=False).head(10))


print("\n================ CORRELATION ================")
corr = df[["rating", "popularity", "clicked"]].corr()
print(corr)

print("\n================ OUTLIERS ================")

Q1 = df["rating"].quantile(0.25)
Q3 = df["rating"].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df["rating"] < Q1 - 1.5 * IQR) | (df["rating"] > Q3 + 1.5 * IQR)]
print("Rating outliers:")
print(outliers)

# Preparing data
X = df[["rating", "popularity", "genre"]]
y = df["clicked"]

# Training
model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("\nModel trained and saved!")
