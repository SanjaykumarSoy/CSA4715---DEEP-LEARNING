import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


wine = load_wine()
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['Target'] = wine.target

x = data.drop('Target', axis=1)
y = data['Target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000, random_state=1)
model.fit(x_train_scaled, y_train)


y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression on Wine Dataset')
plt.show()
