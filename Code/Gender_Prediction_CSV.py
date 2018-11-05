import pandas as pd
from Gender_Prediction_Model import classify_Gender

def prediction(file_name):
    dataset = pd.read_csv(file_name)
    data = dataset.values.tolist()
    
    y_out = []
    for i in data:
        y_out.append(classify_Gender([i]))
    
    res = [['Serial Number','Name','Gender']]
    for i in range(len(data)):
        res.append((data[i][0],data[i][1],y_out[i]))
    return res