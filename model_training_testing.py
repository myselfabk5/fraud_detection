# -*- coding: utf-8 -*-
"""

@author: Abhishek
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier 
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

class model_training_testing():
    
    def models_to_train(classifiers_list=['logistic']):
        
        models = {
            'logistic':LogisticRegression(max_iter = 1000),
            'RandomForest':RandomForestClassifier(random_state=42),
            'XGB': xgb.XGBClassifier(random_state=42),
            'BRF': BalancedRandomForestClassifier(n_estimators=100,  # BRF has inbuilt handling feature for unbalanced class using SMOTE
                                             random_state=42, 
                                             sampling_strategy="all", 
                                             replacement=True,
                                             bootstrap=False),
            'DecisionTree': DecisionTreeClassifier(),
            'KNN': KNeighborsClassifier(),
            'NB': GaussianNB(),
            'GBC': GradientBoostingClassifier(),
            'MLPC': MLPClassifier(),
            'SVC':SVC()
            }
        
        models_for_training = {}
        for j in classifiers_list:
            mod = models[j]
            models_for_training[f'{j}'] = mod
            
        return models_for_training
    

    def model_fitting(x_train,y_train,x_test,y_test,models):
        fitted_models = {}
        models_metrics = [] 
        for key, mod in models.items():
            print(f"Fitting {key} model.")
            mod.fit(x_train,y_train)
            print(f"Fitting {key} model completed.")
            fitted_models[f'{key}'] = mod
            ###################### Predictions on test data #####################
            y_pred = mod.predict(x_test)
            ##################### Model Evaluation ##################
            acc_score = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test,y_pred)
            f1score = f1_score(y_test,y_pred)
            metric_result = {'model':key,'accuracy_score':acc_score,'precision':precision,'recall':recall,'f1_score':f1score}
            models_metrics.append(metric_result)
        final_metrics_df = pd.DataFrame(models_metrics)
        return final_metrics_df, fitted_models 

