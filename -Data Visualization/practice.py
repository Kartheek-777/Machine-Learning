import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    "Hours": [2, 3, 4, 5, 6, 7, 8],
    "Marks": [40, 48, 55, 65, 72, 80, 90]
}

df = pd.DataFrame(data)


plt.plot(df["Hours"], df["Marks"])
plt.show()


plt.scatter(df["Hours"], df["Marks"])
plt.show()


plt.hist(df["Hours"])
plt.show()


sns.scatterplot(data=df, x="Hours", y="Marks")
plt.show()


sns.histplot(df["Marks"])
plt.show()


sns.heatmap(df.corr(), annot=True)
plt.show()