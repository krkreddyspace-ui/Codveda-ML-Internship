import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("house.csv", sep=r"\s+", header=None)


columns = [
    "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
    "DIS","RAD","TAX","PTRATIO","B","LSTAT","PRICE"
]
df.columns = columns


print(df.head())
print(df.isnull().sum())


for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())


X = df.drop("PRICE", axis=1)
y = df["PRICE"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)