import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [18, 20, 21, 22, 25, 30, 32, 35, 45, 50],
    "Salary": [15000, 18000, 19000, 20000, 25000, 35000, 40000, 45000, 70000, 80000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

model = DBSCAN(eps=0.7, min_samples=2)
df["Cluster"] = model.fit_predict(scaled_data)

print("\nData with Clusters:")
print(df)

print("\nNumber of Data Points in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nNoise Points:")
print(df[df["Cluster"] == -1])