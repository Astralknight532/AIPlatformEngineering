# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:46:45 2020

@author: hanan
"""
import pandas as pd

# Defining functions to organize repetitive tasks

# Converts date column to datetime
def dt_convert(data:pd.Series) -> pd.Series: 
    data = pd.to_datetime(data.str.strip(), format = '%Y-%m-%d', errors = 'coerce')
    return data

# Converts the passed string values into type float64 - extracts strings using regex
def float_convert(data:pd.Series) -> pd.Series:
    data = data.str.extract('(\d*\.\d*)').astype('float64')
    return data