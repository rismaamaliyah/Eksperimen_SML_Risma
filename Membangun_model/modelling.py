import dagshub
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dagshub.init(repo_owner="rismaamaliyah", repo_name="Eksperimen_SML_Risma", mlflow=True)
mlflow.set_experiment("Adult Income Prediction")

mlflow.autolog(log_models=True)

train_df = pd.read_csv("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing.csv")
test_df = pd.read_csv("Membangun_model/adult_income_dataset_preprocessing/adult_income_dataset_preprocessing_test.csv")

X_train = train_df.drop(columns=["income"])
y_train = train_df["income"]

X_test = test_df.drop(columns=["income"])
y_test = test_df["income"]

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))