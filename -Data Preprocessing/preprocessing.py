import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


data = {
    "Age": [20, 21, np.nan, 23, 22, 24],
    "Salary": [25000, 30000, 28000, np.nan, 35000, 40000],
    "Department": ["CSE", "ECE", "CSE", "AIML", "ECE", "CSE"],
    "Experience": [1, 2, 1, 3, 2, 4]
}

df = pd.DataFrame(data)

print(df)


# Missing values

print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

print(df)


# Encoding categorical data

encoder = LabelEncoder()

df["Department"] = encoder.fit_transform(df["Department"])

print(df)


# Separating features and target

X = df[["Age", "Salary", "Department"]]
y = df["Experience"]

print(X)
print(y)


# Splitting data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:")
print(X_train)

print("Testing data:")
print(X_test)


# Feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaled training data:")
print(X_train)

print("Scaled testing data:")
print(X_test)