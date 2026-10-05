import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 48, 55, 62, 68, 75, 82, 90, 95]
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

predictions = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", predictions)

hours = [[7.5]]

prediction = model.predict(hours)

print("Predicted marks:", prediction[0])

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)


plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression")
plt.show()