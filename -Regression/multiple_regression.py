import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
     "Age": [21, 22, 23, 24, 25, 26, 27, 28],
    "Projects": [1, 2, 2, 3, 4, 4, 5, 6],
    "Salary": [25000, 30000, 35000, 42000, 50000, 58000, 65000, 75000]
}

df = pd.DataFrame(data)

X = df[["Experience", "Age", "Projects"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Actual salaries:")
print(list(y_test))

print("Predicted salaries:")
print(predictions)


new_employee = [[3, 21, 3]]

salary = model.predict(new_employee)

print("Predicted salary:", salary[0])