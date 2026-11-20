import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
print("First 5 rows of the dataset:")
print(data.head())

# 2. Define features and target
X = data[['sepal length (cm)']]
y = data['sepal width (cm)']

# 3. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions and calculate MSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error on test set: {mse:.2f}')

# 6. Sort X_test and y_pred so plt.plot draws a clean, continuous line
sort_indices = X_test['sepal length (cm)'].argsort()
X_test_sorted = X_test.iloc[sort_indices]
y_pred_sorted = y_pred[sort_indices]

# 7. Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='r', marker='*', label='Actual')
plt.plot(X_test_sorted, y_pred_sorted, color='y', linewidth=3, label='Predicted (Regression Line)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Linear Regression: Sepal Width vs Sepal Length')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 8. Predict sepal width for a new sample
new_sample = pd.DataFrame([[5.0]], columns=['sepal length (cm)'])
predicted_width = model.predict(new_sample)
print(f'The predicted sepal width for sepal length {new_sample.values[0][0]} cm is {predicted_width[0]:.2f} cm')
