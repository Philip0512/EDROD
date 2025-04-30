import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
from model import EDROD

def main(args):
    # Build the full path to the dataset
    path = f'{args.path_prefix}{args.data_name}_x.csv'
    path_label = f'{args.path_prefix}{args.data_name}_y.csv'

    # Load feature data 
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    # Initialize and fit the EDROD anomaly detector
    detector = EDROD(n_neighbors=args.n_neighbors, metric="euclidean")
    detector.fit(data)

    # Get anomaly scores
    edr_score = detector.decision_scores_

    # Load ground truth labels 
    df = pd.read_csv(path_label)
    y_true = df.iloc[:, 0]

    # Normalize scores for ROC computation
    y_pred = edr_score / np.sum(edr_score)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Print AUC result
    print(f"AUC: {roc_auc:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run EDROD on a synthetic dataset and compute AUC.')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the dataset')
    parser.add_argument('--n_neighbors', type=int, default=20,
                        help='Number of neighbors for EDROD')
    parser.add_argument('--path_prefix', type=str, default='dataset/',
                        help='Prefix path to dataset file')
    args = parser.parse_args()
    main(args)
