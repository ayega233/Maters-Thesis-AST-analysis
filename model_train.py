from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV

class ModelTrainer:

    def __init__(self):  
        self.train_count = 0

    def train(self,train_repos):
        traind_data = []
        for  re in train_repos:
            train_df = pd.read_csv(f"train/{re}_feature.csv")
            traind_data.append(train_df)
        
        combined_df = pd.concat(traind_data, ignore_index=True)

        X_train = combined_df.drop(columns=["is_defective"])
        self.train_count =combined_df.shape[0]
        X_train = X_train.drop(columns=["num_changes"])
        y_train = combined_df["is_defective"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
       
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
    def test(self,repo):

        test_df = pd.read_csv(f"test/{repo}_feature.csv")    
        val_df = pd.read_csv(f"val/{repo}_feature.csv") 
       
        X_val = val_df.drop(columns=["is_defective"])
        X_val = X_val.drop(columns=["num_changes"])
        y_val = val_df["is_defective"]

        X_test = test_df.drop(columns=["is_defective"])
        X_test = X_test.drop(columns=["num_changes"])
        y_test = test_df["is_defective"]

        

        y_val_pred = self.model.predict(X_val)

        report_validation = classification_report(y_val, y_val_pred, output_dict=True)
        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        
        y_test_pred = self.model.predict(X_test)

        report = classification_report(y_test, y_test_pred, output_dict=True)

        print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
        

        results= {
            "repo":repo,
            "training_set":self.train_count,
            "validation_set":val_df.shape[0],
            "test_set":test_df.shape[0],
            "valiadation_accuracy": round(accuracy_score(y_val, y_val_pred),4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred),4),   
             **self.best_params
        }
        for class_label, metrics in report.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"test_{class_label}_{metric}"] = round(value, 4)

        for class_label, metrics in report_validation.items():
            if class_label != 'accuracy':  
                for metric, value in metrics.items():
                    results[f"validation_{class_label}_{metric}"] = round(value, 4)            
        return results


if __name__ == "__main__":
    #,"django"
    #repos = ["airflow","cpython","scikit-learn","celery","transformers","localstack","spaCy","yolov5","numpy","jax","poetry","openpilot","black","lightning","pandas","sentry","ray","redash","scrapy","pipenv"]
   #trainer = ModelTrainer("scikit-learn")
    #trainer = ModelTrainer("cpython")
    #trainer.train()
    #repos = ["airflow","cpython","scikit-learn","celery","transformers","localstack","spaCy","yolov5","numpy","jax","poetry","openpilot"]
    repos = ["airflow","cpython","scikit-learn","celery"]
    
    accuracy_data = []  
    trainer = ModelTrainer()
    trainer.train(["airflow","cpython","scikit-learn","celery"])
    for repo in repos:          
        accuracy_data.append(trainer.test(repo))
    df_f = pd.DataFrame(accuracy_data)
    df_f.to_csv(f"accuracy_combined.csv", index=False)       
