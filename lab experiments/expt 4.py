import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split into training and testing sets (using a larger split to see overfitting behaviors clearly)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define different depths to evaluate
depths = range(1, 15)
train_accs = []
test_accs = []

for depth in depths:
    # Train a decision tree with a specific depth constraint
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on both training and testing splits
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    train_accs.append(train_acc)
    test_accs.append(test_acc)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accs, marker='o', label='Training Accuracy', color='blue', lw=2)
plt.plot(depths, test_accs, marker='s', label='Testing Accuracy', color='red', lw=2)

plt.xlabel('Tree Depth (Model Complexity)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Underfitting vs. Overfitting (Decision Tree on Wine Dataset)', fontsize=14)
plt.xticks(depths)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()
