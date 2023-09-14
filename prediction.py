import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def categorical_enc(select,optionslist):
    idx = optionslist.index(select)
    area_enc = [0]*len(optionslist)
    area_enc[idx] = 1
    return area_enc

def ordinal_enc_day(day):
    days_dict = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, "Friday":6, "Saturday":7}
    return days_dict[day]

def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    out = model.predict(data)
    severity = {1: 'Slight Injury', 2: 'Serious Injury', 3 : 'Fatal Injury'}
    return severity[out[0]]
    