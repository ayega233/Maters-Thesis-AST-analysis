from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class ModelTrainer:

    def __init__(self, repo):       
        self.repo = repo

    def train(self):
        
        train_df = pd.read_csv(f"train/{self.repo}_feature.csv")  
        test_df = pd.read_csv(f"test/{self.repo}_feature.csv")    
        val_df = pd.read_csv(f"val/{self.repo}_feature.csv") 


        X_train = train_df.drop(columns=["is_defective"])
        X_train = X_train.drop(columns=["num_changes"])
        y_train = train_df["is_defective"]

        X_val = val_df.drop(columns=["is_defective"])
        X_val = X_val.drop(columns=["num_changes"])
        y_val = val_df["is_defective"]

        X_test = test_df.drop(columns=["is_defective"])
        X_test = X_test.drop(columns=["num_changes"])
        y_test = test_df["is_defective"]
       
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }


        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X_train)
        X_scaled_df =pd.DataFrame(X_scaled, columns=X_train.columns)
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_scaled_df, y_train)

        X_test_scaled = scaler.transform(X_test)
        X_test_scaled_df =pd.DataFrame(X_test_scaled, columns=X_test.columns)
       
       
        best_model = grid_search.best_estimator_
        y_test_pred = best_model.predict(X_test_scaled_df)

        feature_importance = best_model.feature_importances_

        # Create DataFrame
        feature_names = X_train.columns
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

        # Sort by importance
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
        print(importance_df)


if __name__ == "__main__":
    #,"django"
    #repos = ["airflow","cpython","scikit-learn","celery","transformers","localstack","spaCy","yolov5","numpy","jax","poetry","openpilot","black","lightning","pandas","sentry","ray","redash","scrapy","pipenv"]
   #trainer = ModelTrainer("scikit-learn")
    #trainer = ModelTrainer("cpython")
    #trainer.train()
    repos = ["airflow"]
    
    accuracy_data = []  
    for repo in repos:  
        trainer = ModelTrainer(repo)
        accuracy_data.append(trainer.train())
    df_f = pd.DataFrame(accuracy_data)
    df_f.to_csv(f"accuracy_features.csv", index=False)       
