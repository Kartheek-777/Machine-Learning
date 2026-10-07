import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = {
    "age": [20, 22, 25, 27, 30, 32, 35, 38, 40, 42],
    "income": [20000, 25000, 30000, 32000, 40000, 45000, 50000, 55000, 60000, 65000],
    "purchased": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "income"]]
y = df["purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))


new_customer = [[28, 35000]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Prediction: Customer may purchase")
else:
    print("Prediction: Customer may not purchase")