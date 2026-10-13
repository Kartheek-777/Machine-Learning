import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [20, 22, 25, 28, 35, 38, 40, 45, 50, 55],
    "Income": [15000, 18000, 22000, 25000, 40000, 45000, 50000, 65000, 75000, 85000]
}

df = pd.DataFrame(data)

print("Customer Data:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(scaled_data)

print("\nCustomer Groups:")
print(df)

print("\nCluster Summary:")
print(df.groupby("Cluster")[["Age", "Income"]].mean().round(2))

print("\nEnter Customer Details:")
age = int(input("Enter age: "))
income = float(input("Enter annual income: "))

customer = scaler.transform([[age, income]])
cluster = model.predict(customer)

print(f"\nCustomer belongs to Cluster {cluster[0]}")