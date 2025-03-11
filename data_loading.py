import pandas as pd 
import numpy as np 
import os

def load_data(datapath):
    df_dict = {}
    for filename in os.listdir(datapath):
        if filename.endswith('.csv'):
            df_dict[filename] = pd.read_csv(os.path.join(datapath, filename))
    
    data = {}
    for i in range(3):
        tf_key = f"TF{i}"
        X_train = np.array(df_dict[f"Xtr{i}.csv"]['seq'])
        X_test  = np.array(df_dict[f"Xte{i}.csv"]['seq'])
        
        y_train = df_dict[f"Ytr{i}.csv"]['Bound'].values.ravel()
        # Convert zeros to -1 for classification
        y_train = np.where(y_train == 0, -1, y_train)
        
        data[tf_key] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train
        }
    return data
    