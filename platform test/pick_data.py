import pandas as pd

data = pd.read_csv("esg_data.csv")

X = data.drop(columns=["ESG_score", "Company"])
y = data["ESG_score"]
print(X)
print(y)
