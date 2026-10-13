import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [18, 20, 22, 25, 30, 32, 35, 40, 45, 50],
    "Salary": [15000, 18000, 20000, 25000, 35000, 40000, 45000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

model = AgglomerativeClustering(n_clusters=3)
df["Cluster"] = model.fit_predict(scaled_data)

print("\nData with Clusters:")
print(df)

print("\nNumber of Data Points in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())