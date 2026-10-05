import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90],
    "Age": [18, 19, 20, 21, 22, 23, 24, 25]
}

df = pd.DataFrame(data)

print(df)


# Line Plot

plt.plot(df["Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()


# Scatter Plot

plt.scatter(df["Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Scatter Plot")
plt.show()


# Bar Chart

subjects = ["Python", "SQL", "ML", "NLP"]
students = [40, 35, 25, 20]

plt.bar(subjects, students)
plt.xlabel("Subjects")
plt.ylabel("Students")
plt.title("Students per Subject")
plt.show()


# Histogram

plt.hist(df["Marks"])
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Marks Distribution")
plt.show()


# Box Plot

plt.boxplot(df["Marks"])
plt.title("Box Plot")
plt.show()


# Seaborn Scatter Plot

sns.scatterplot(data=df, x="Hours", y="Marks")
plt.show()


# Seaborn Histogram

sns.histplot(df["Marks"])
plt.show()


# Correlation

print(df.corr())


sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()