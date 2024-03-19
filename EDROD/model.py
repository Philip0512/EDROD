import warnings
from numpy.linalg import cholesky
import numpy as np
from sklearn.neighbors import KDTree
from scipy.linalg import sqrtm
from utils import variable_kernel_density_estimation, EDR, find_k_nearest_neighbors

class EDROD:
    '''
    Unsupervised Outlier Detection Method using Entropy Density Raio
    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors.
    metric : str or callable, {'mahalanobis', 'euclidean', 'canberra', 'correlation', 'dice'}
        the distance metric to use for the bandwidth of variable kernel density estimation.
        The default metric is mahalanobis.
        You can choose different metrics to suit different data.
        More metrics:
        - 'chebyshev'
        - 'seuclidean'
        - 'cosine'
        - 'cityblock'
        - 'jaccard'


    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        The outlier scores of input samples. The higher, the more abnormal.

    n_neighbors : int
        The actual number of neighbors.

    Examples
    --------
    from model import EDROD

    detector = EDROD(n_neighbors=20)
    detector.fit(X)
    y_outlier_score = detector.decision_scores_
    # y_outlier_score is the outlier scores of samples in X.
    # You can use it to calculate AUC.
    '''
    def __init__(self, *, n_neighbors=20, metric="euclidean"):
        self.n_neighbors=n_neighbors
        self.metric=metric

    def fit(self, X, y=None):
        """
        Fit the model using X as training data.
        And predict the outlier score of X.

        :param X:
            Supposed to be numpy.ndarray.
        :param y:
            Ignored
            Not used, present for API consistency by convention.
        :return:
            self : object
        """
        n_samples = X.shape[0]
        if self.n_neighbors > n_samples:
            warnings.warn("n_neighbors (%s) is greater than the "
                          "total number of samples (%s). n_neighbors "
                          "will be set to (n_samples - 1) for estimation."
                          % (self.n_neighbors, n_samples))

        density = variable_kernel_density_estimation(X)
        X_normalization = normalization_for_KDTree(X)
        # Construct KDTree
        tree = KDTree(X_normalization)
        EDR_Score = np.array([])
        for point in X_normalization:
            md_indices = find_k_nearest_neighbors(point.reshape(1, -1), self.n_neighbors, tree)
            md_indices = md_indices.flatten()
            md_nearst_nb = X[md_indices]
            EDR_Score = np.append(EDR_Score, EDR(density, md_indices))

        self.decision_scores_ = EDR_Score
        return self




def normalization_for_KDTree(X):
    # Calculate the covariance matrix and its inverse square root.
    cov_matrix = np.cov(X, rowvar=False)
    sqrt_inv_cov = np.linalg.inv(sqrtm(cov_matrix))

    # Transform the data using the inverse square root of the covariance matrix.
    transformed_data = (X - np.mean(X, axis=0)).dot(sqrt_inv_cov)

    return transformed_data
