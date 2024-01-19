import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from model import EDROD

if __name__ == '__main__':

    # Load Dataset
    file='1'
    path = 'dataset/synthetic/synthetic_test_'+f"{file}"+'_process.csv'
    data = np.genfromtxt(path, delimiter=',',
                        skip_header=1,usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))


    n_neighbors = 20
    detector = EDROD(n_neighbors=n_neighbors, metric="mahalanobis")
    detector.fit(data)
    EDR_Score = detector.decision_scores_


    #Calculate AUC Value
    # Load Label
    df = pd.read_csv(path)
    data_label = df.iloc[:, 10]
    y_true = data_label
    y_pred = EDR_Score / np.sum(EDR_Score)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    print(roc_auc)
