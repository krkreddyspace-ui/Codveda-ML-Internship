import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

df = pd.read_csv("iris.csv")

X = df[["petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_linear = SVC(kernel="linear", probability=True)
model_linear.fit(X_train, y_train)

model_rbf = SVC(kernel="rbf", probability=True)
model_rbf.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)
y_pred_rbf = model_rbf.predict(X_test)

print("Linear Kernel Accuracy:", accuracy_score(y_test, y_pred_linear))
print("\nLinear Classification Report:\n", classification_report(y_test, y_pred_linear))

print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("\nRBF Classification Report:\n", classification_report(y_test, y_pred_rbf))

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = model.predict(grid)
    Z = pd.factorize(Z)[0]
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pd.factorize(y)[0])
    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.title(title)
    plt.show()

plot_decision_boundary(model_linear, X, y, "SVM Linear Kernel")
plot_decision_boundary(model_rbf, X, y, "SVM RBF Kernel")