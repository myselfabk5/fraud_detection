# -*- coding: utf-8 -*-
"""

@author: Abhishek
"""
class data_loading():
    def load_data(path):
        import pandas as pd
        data = pd.read_csv(path)
        return data
