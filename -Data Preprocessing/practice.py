import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


data = {
    "Age": [21, 22, np.nan, 24, 23],
    "Marks": [75, 82, 90, np.nan, 68],
    "City": ["Hyderabad", "Khammam", "Hyderabad", "Warangal", "Khammam"]
}

df = pd.DataFrame(data)

print(df)

print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

print(df)


encoder = LabelEncoder()

df["City"] = encoder.fit_transform(df["City"])

print(df)


X = df[["Age", "Marks", "City"]]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)