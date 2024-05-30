from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the breast cancer dataset
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels,
test_size=0.2, random_state=42)
# Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()
# Train the classifier on the training data
gnb.fit(X_train, y_train)
# Predict the labels for test features
y_pred = gnb.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)