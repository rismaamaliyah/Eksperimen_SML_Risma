import dagshub
import mlflow
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

dagshub.init(repo_owner="rismaamaliyah", repo_name="Eksperimen_SML_Risma", mlflow=True)
mlflow.set_experiment("Adult Income Prediction Tuning")

train_df = pd.read_csv("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing.csv")
test_df = pd.read_csv("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing_test.csv")

X_train = train_df.drop(columns=["income"])
y_train = train_df["income"]

X_test = test_df.drop(columns=["income"])
y_test = test_df["income"]

param_grid = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

with mlflow.start_run():
    # Log best parameters and metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("best_C", grid.best_params_["C"])
    mlflow.log_param("best_solver", grid.best_params_["solver"])
    mlflow.log_metric("accuracy", acc)
    
    # Save the best model
    joblib.dump(best_model, "adult_income_model_tuned.pkl")
    mlflow.log_artifact("adult_income_model_tuned.pkl", artifact_path="models")
    
    # Log preprocessing data
    mlflow.log_artifact("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing.csv", artifact_path="data")
    
    # Additional artifact logs
    mlflow.log_artifact("Membangun_model/screenshoot_dashboard.jpg", artifact_path="screenshoot")
    mlflow.log_artifact("Membangun_model/screenshoot_artifact.jpg", artifact_path="screenshoot")
    mlflow.log_artifact("Membangun_model/screenshoot_artifact_2.jpg", artifact_path="screenshoot")
    mlflow.log_artifact("Membangun_model/screenshoot_artifact_charts.jpg", artifact_path="screenshoot")