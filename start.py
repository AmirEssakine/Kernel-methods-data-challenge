import numpy as np
import pandas as pd
from data_loading import load_data
from kernels import *
from SVM import C_SVM

# Set Parameters
path = r"C:\Users\Essakine\Downloads\CT\Kernel methods\data-challenge-kernel-methods-2024-2025"
out_path = "./"
C = 1.

def score(pred, labels):
    return np.mean(pred == labels)

kernel = ConvNgramKernel(gamma=32.0,n=4,k=9,M=4096).kernel
val_ratio = 0.2
number_of_runs = 9
method = 'CSVM'

data = load_data(path)
print("Size of dataset: ", data['TF2']['X_train'].shape, data['TF2']['X_test'].shape)

print('************** ' + str(method) + ' *****************')
algo = svm = C_SVM(kernel,solver='BFGS')

best_pred_tests = {}
best_scores = {f'TF{j}': 0 for j in range(3)}

for i in range(number_of_runs):
    for j in range(3):
        dataset_key = f'TF{j}'
        X_full = data[dataset_key]['X_train']
        y_full = data[dataset_key]['y_train']
        X_test = data[dataset_key]['X_test']
        
        n = X_full.shape[0]
        p = int(val_ratio * n) 
        idx_val = np.random.choice(n, p, replace=False)
        idx_train = np.setdiff1d(np.arange(n), idx_val)

        X_train, y_train = X_full[idx_train], y_full[idx_train]
        X_val, y_val = X_full[idx_val], y_full[idx_val]
        
        print('--Begin fitting')
        algo.fit(X_train, y_train)
        pred_train = algo.predict(X_train)
        print('Train dataset accuracy:', score(pred_train, y_train))
        
        pred_val = algo.predict(X_val)
        val_score = score(pred_val, y_val)
        print("Score on validation set:", val_score)
        
        pred_test = algo.predict(X_test)
        
        # Store best prediction based on validation score
        if val_score > best_scores[dataset_key]:
            best_scores[dataset_key] = val_score
            best_pred_tests[dataset_key] = pred_test

# Saving results
saving = True
filename = "best_predictions"

if saving:
    final_preds = np.concatenate([best_pred_tests[f'TF{j}'] for j in range(3)])
    data_id = np.arange(final_preds.shape[0])
    data_val = np.where(final_preds == -1, 0, 1)
    
    df = pd.DataFrame({"Id": data_id, "Bound": data_val})
    df.to_csv(out_path + filename + ".csv", index=False)
    print(f"Best predictions saved to {out_path + filename}.csv")
