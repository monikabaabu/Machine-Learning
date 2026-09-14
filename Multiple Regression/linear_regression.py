import pandas as pd

df = pd.read_csv("data (2).csv")
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df[df["price"] == 0])
print("Number of zero-price houses:", (df["price"] == 0).sum())
df = df[df["price"] > 0]
print(df.shape)
X = df[["sqft_living"]]

y = df["price"]
print(X.head())
print(y.head())
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price")
plt.title("Living Area vs House Price")
plt.show()

print(df["sqft_living"].corr(df["price"]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
prediction = model.predict([[2000]])

print("Predicted price:", prediction[0])

print("Actual price:", y_test.iloc[0])
print("Predicted price:", model.predict(X_test.iloc[[0]])[0])

error = y_test.iloc[0] - model.predict(X_test.iloc[[0]])[0]

print("Prediction error:", error)
y_pred = model.predict(X_test)
print("Number of predictions:", len(y_pred))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Root Mean Squared Error:", rmse)

r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)

plt.xlabel("Living Area (sqft)")
plt.ylabel("Price")
plt.title("Linear Regression: Living Area vs Price")

plt.show()

residuals = y_test - y_pred
print(residuals.head())

plt.scatter(X_test, residuals)
plt.axhline(y=0, color="red")

plt.xlabel("Living Area (sqft)")
plt.ylabel("Residual")
plt.title("Residuals vs Living Area")

plt.show()
print(df.corr(numeric_only=True)["price"].sort_values(ascending=False))

X = df[["sqft_living", "bathrooms", "bedrooms", "view", "waterfront"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

print(y_pred[:5])


mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)

print("R²:", r2)
print(df[["bedrooms", "sqft_living"]].corr())
print(df["bedrooms"].corr(df["price"]))