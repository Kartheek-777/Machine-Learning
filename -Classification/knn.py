import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = {
    "age": [18, 20, 22, 25, 28, 30, 32, 35, 38, 40],
    "salary": [20000, 22000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000],
    "purchased": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "salary"]]
y = df["purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Predictions:", y_pred)
print("Actual:", y_test.values)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


new_customer = [[29, 38000]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Prediction: Customer may purchase")
else:
    print("Prediction: Customer may not purchase")