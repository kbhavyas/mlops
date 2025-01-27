import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Iris dataset from the CSV file
iris = pd.read_csv("data/iris_data.csv")

# Separate features (X) and the target variable (y)
X = iris.drop(columns=["target"])  # Drop the target column to get features
y = iris["target"]  # Target variable

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

''' Loop through different values of the 'max_depth' parameter for experimentation'''
for max_depth in [2, 3, 4]:
    # Start an MLflow experiment run
    with mlflow.start_run():
        # Initialize and train the Decision Tree Classifier with the current 'max_depth'
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, y_train)

        # Predict on the test set and evaluate the model's accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log the hyperparameter ('max_depth') and evaluation metric ('accuracy') to MLflow
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(clf, "decision_tree_model")

        # Print the results for the current experiment run
        print(f"Run completed for max_depth={max_depth}, accuracy={accuracy:.4f}")
