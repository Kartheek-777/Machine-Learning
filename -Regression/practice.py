import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Marks": [30, 38, 45, 52, 60, 68, 75, 84, 92]
}

df = pd.DataFrame(data)

X = df[["StudyHours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", predictions)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))

new_data = [[10]]

prediction = model.predict(new_data)

print("Predicted marks:", prediction[0])