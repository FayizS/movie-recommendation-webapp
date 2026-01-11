import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
import joblib

conn = sqlite3.connect("clicks.db")
df = pd.read_sql("SELECT * FROM clicks", conn)

X = df[["rating"]]
y = df["clicked"]

model = LogisticRegression()
model.fit(X,y)

joblib.dump(model,"model.pkl")
print("Model trained")