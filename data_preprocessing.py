# -*- coding: utf-8 -*-
"""

@author: Abhishek
"""
# Need to provide the entire dataset and target(y) column and all the relevant columns of the data including target
# target will be in string (example: "target"), columns_to_consider will be list of columns to consider
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class data_preprocessing():
    def data_preprocess(data,target,columns_to_consider, test_size=0.20):

        required_data = data[columns_to_consider]
        Y = required_data[target]
        X = required_data.drop([target], axis = 1)
        
        cat_columns = X.select_dtypes(include=['object']).columns
        num_columns = X.select_dtypes(exclude=['object']).columns
        
        ############ one hot encoding for categorical columns ###############
        cat_df = pd.get_dummies(X[cat_columns], drop_first=True, dtype=float)
        ############# combining both dataframes #############
        cat_df = cat_df.reset_index(drop=True)
        num_df = X[num_columns]
        num_df = num_df.reset_index(drop = True)
        
        combined_df = pd.concat([cat_df,num_df], axis = 1)
        
        ###################### train test split #############################
        x_train,x_test,y_train,y_test = train_test_split(combined_df,Y, test_size = test_size, random_state = 42)
        
        ############ standardising the numeric features #################
        scaler = StandardScaler()
        x_train[num_columns] = scaler.fit_transform(x_train[num_columns])
        x_test[num_columns] = scaler.fit_transform(x_test[num_columns])
        
        return x_train,x_test,y_train,y_test
    
    def shape_data(data, target):
        cols = data.columns
        print('Data Shape:', data.shape)
        print('Columns in the data:', cols) 
        df_list = []
        for col in cols:
            unique_val = data[col].nunique()
            one_row = {'column_name':col, 'unique_value':unique_val}
            df_list.append(one_row)
        df = pd.DataFrame(df_list)
        # plotting the target
        plt.hist(data[target])
        plt.title(f'Histogram for {target}')
        plt.show();
        return df
        
