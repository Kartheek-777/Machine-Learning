import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    "Hours": [2, 3, 4, 5, 6, 7, 8, 9],
    "Marks": [40, 45, 52, 60, 68, 75, 82, 90]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

predictions = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", predictions)

hours = [[10]]

prediction = model.predict(hours)

print("Predicted marks:", prediction[0])