import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Salary": [25, 29, 35, 44, 56, 70, 87, 108]
}

df = pd.DataFrame(data)

X = df[["Experience"]]
y = df["Salary"]

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", predictions)


new_experience = [[6.5]]

new_experience = poly.transform(new_experience)

prediction = model.predict(new_experience)

print("Predicted salary:", prediction[0])


plt.scatter(X, y)

plt.plot(
    X,
    model.predict(poly.transform(X))
)

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.show()