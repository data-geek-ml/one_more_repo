import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split,ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("dibetes.csv")

X = data.drop(columns=["Outcome"])

Y = data["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train =  Y_train.values.reshape(-1,1)
Y_test =  Y_test.values.reshape(-1,1)
X_train= X_train.values
X_test = X_test.values


lr = LogisticRegression()

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200]
}

param_grid = list(ParameterGrid(param_grid))

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('Dev Expermiment Prediction')



best_accuracy = 0
best_params = None
best_model = None

for params in param_grid:
    with mlflow.start_run():
        # Create the model with the current parameters
        model = LogisticRegression(**params)

        # Train the model
        model.fit(X_train, Y_train)

        # Make predictions
        Y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, Y_pred)

        # Log the model
        model_uri = mlflow.sklearn.log_model(model, "model")

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Update the best model if the current one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model
            best_model_uri = model_uri

# Print the best parameters and accuracy
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)


best_model_uri_ext = best_model_uri.model_uri

model_name = "NewModel"

result = mlflow.register_model(best_model_uri_ext, model_name)

print("training complete and model regsitration sucesful")
