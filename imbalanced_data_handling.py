# -*- coding: utf-8 -*-
"""

@author: Abhishek
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

class imbalanced_data_handling():
    def imb_data_handling(x_train, y_train):
        smote = SMOTE(random_state = 42)
        X_train_smote, y_train_smote = smote.fit_resample(x_train.to_numpy(),np.array(y_train))
        X_train_smote_df = pd.DataFrame(X_train_smote, columns = x_train.columns)
        
        print('Shape of training data(X) after applying SMOTE is:', X_train_smote_df.shape)
        print('Shape of training data(Y) after applying SMOTE is:', y_train_smote.shape)
        print('Value counts for training data(Y) after applying SMOTE:', pd.Series(y_train_smote).value_counts())
        print('Value counts for training data(Y) before applying SMOTE:', pd.Series(y_train).value_counts())
    
        return X_train_smote_df, pd.Series(y_train_smote)
    

