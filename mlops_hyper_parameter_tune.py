import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for RandomForestClassifier
    n_estimators = trial.suggest_int('n_estimators', 11, 200)  # Number of trees in the forest
    max_depth = trial.suggest_int('max_depth', 2, 20)  # Maximum depth of the tree, reduced range for small datasets
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  # Minimum samples required to split an internal node

    # Train the Random Forest model with the suggested hyperparameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42  # Set random seed for reproducibility
    )
    model.fit(X_train, y_train)  # Fit the model on the training data
    y_pred = model.predict(X_test)  # Predict on the test data
    return accuracy_score(y_test, y_pred)  # Return accuracy as the objective metric

# Run Optuna hyperparameter optimization to maximize the accuracy
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Run optimization for 20 trials

# Save the hyperparameter tuning results as a DataFrame
results_df = study.trials_dataframe()  # Convert the results into a DataFrame for analysis
results_df.to_csv("hyperparameter_tuning_report.csv", index=False)  # Save the results as a CSV file

# Train the best model with the best found hyperparameters
best_params = study.best_params  # Get the best parameters from the optimization process
print("Best parameters:", best_params)

# Train the model with the best hyperparameters
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate the model's accuracy on the test data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)

# Optional: Save the results DataFrame for further analysis
results_df.head()  # Display the top rows of the results DataFrame

