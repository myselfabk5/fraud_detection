# -*- coding: utf-8 -*-
"""

@author: Abhishek
"""

# main.py
import os
os.chdir(r"C:\Users\DELL\OneDrive\Data-Science\Data_Science_Projects_Data\loan_fraud_detection\fraud_detection")
from data_loading import data_loading
from data_preprocessing import  data_preprocessing
from model_training_testing import model_training_testing
from imbalanced_data_handling import imbalanced_data_handling


def main():
    # Load your data
    print('Loading Data')
    data = data_loading.load_data(r"C:\Users\DELL\OneDrive\Data-Science\Data_Science_Projects_Data\loan_fraud_detection\loan_fraud_detection_dataset.csv")
    print('Data Loaded')

    # Data perprocessing and feature engineering
    target = 'fraud'
    columns_to_consider = ['age','gender','merchant','category','amount','fraud']
    test_size = 0.25
    
    ## Data shape and value counts
    print('Data preprocessing started')
    data_shape = data_preprocessing.shape_data(data, target)
    print(data_shape)
    
    x_train,x_test,y_train,y_test = data_preprocessing.data_preprocess(data,target,columns_to_consider, test_size=test_size)
    print('Data preprocessing completed')
    # Train the model
    # logistic is default
    print('Model training without SMOTE started')
    classifiers_list = ['logistic','DecisionTree'] # 'logistic','DecisionTree','RandomForest', 'XGB', 'BRF' (Balanced random forest), 'DecisionTree','KNN', 'NB' (Naive Bayes), 'GBC', 'MLPC', 'SVC'
    models = model_training_testing.models_to_train(classifiers_list)
    final_metrics_df, fitted_models = model_training_testing.model_fitting(x_train,y_train,x_test,y_test,models)
    print('Model training without SMOTE completed')
    # Handling imbalanced data 
    print('Model training with SMOTE started')
    x_train_smote, y_train_smote = imbalanced_data_handling.imb_data_handling(x_train, y_train)
    final_metrics_df_smote, fitted_models_smote = model_training_testing.model_fitting(x_train_smote,y_train_smote,x_test,y_test,models)
    print('Model training with SMOTE completed')
    print('Without Smote Outcome:',final_metrics_df)
    print('With Smote Outcome:',final_metrics_df_smote)
    return final_metrics_df,final_metrics_df_smote,fitted_models,fitted_models_smote


if __name__ == "__main__":
    
    final_metrics_df,final_metrics_df_smote,fitted_models,fitted_models_smote = main()
