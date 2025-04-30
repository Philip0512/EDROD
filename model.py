import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from scipy.linalg import sqrtm
from sklearn.neighbors import KDTree

class EDROD:
    def __init__(self, n_neighbors=20, metric="euclidean", bandwidth=0.5):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.bandwidth = bandwidth
        self.data = None
        self.transformed_data = None
        self.kdtree = None
        self.density = None
        self.decision_scores_ = None

    def fit(self, data):
        self.data = data
        self._preprocess_data()
        self._build_kdtree()
        self._compute_density()
        self._compute_edr_scores()

    def _preprocess_data(self):
        # Apply Mahalanobis-like transformation
        cov_matrix = np.cov(self.data, rowvar=False)
        sqrt_inv_cov = np.linalg.inv(sqrtm(cov_matrix))
        self.transformed_data = (self.data - np.mean(self.data, axis=0)).dot(sqrt_inv_cov)

    def _build_kdtree(self):
        self.kdtree = KDTree(self.transformed_data)

    def _compute_density(self):
        n, _ = self.data.shape
        distances = cdist(self.data, self.data, self.metric)
        local_density = np.mean(distances, axis=1)
        density_estimates = np.zeros(n)

        for i in range(n):
            variable_bandwidth = self.bandwidth * local_density[i]
            kernels = self._gaussian_kernel(distances[i] / variable_bandwidth)
            density_estimates[i] = np.sum(kernels) / (n * variable_bandwidth)

        self.density = density_estimates

    def _compute_edr_scores(self):
        scores = []
        for point in self.transformed_data:
            neighbors = self._find_k_nearest_neighbors(point)
            local_density = self.density[neighbors]
            local_density = local_density / np.sum(local_density)
            point_entropy = entropy(local_density)
            score = point_entropy / local_density[0]  # EDR
            scores.append(score)

        self.decision_scores_ = np.array(scores)

    def _find_k_nearest_neighbors(self, point):
        distances, indices = self.kdtree.query(point.reshape(1, -1), self.n_neighbors)
        return indices.flatten()

    @staticmethod
    def _gaussian_kernel(u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)