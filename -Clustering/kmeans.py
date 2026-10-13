import pandas as pd
from sklearn.cluster import KMeans
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

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(scaled_data)

df["Cluster"] = model.labels_

print("\nData with Clusters:")
print(df)

print("\nCluster Centers:")
print(scaler.inverse_transform(model.cluster_centers_))