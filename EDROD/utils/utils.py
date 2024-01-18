import time
import numpy as np
from scipy.spatial.distance import cdist

# Calculate the function time
def time_cost(f):
    def run_time(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        run_times = time.time() - start
        print("Run time：%.6f s" % (run_times))
        return res

    return run_time



def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)


def variable_kernel_density_estimation(data, bandwidth=1.0):
    n, d = data.shape
    density_estimates = np.zeros(n)

    # Calculate the distance matrix for all points.
    # The method of distance measurement can be altered,
    #   and specific options can be found in the cdist documentation.
    distances = cdist(data, data, 'mahalanobis')

    # Calculate the bandwidth feature for each point.
    local_density = np.mean(distances, axis=1)

    # For each point, adjust the bandwidth based on its local density.
    for i in range(n):
        variable_bandwidth = bandwidth * local_density[i]
        kernels = gaussian_kernel(distances[i] / variable_bandwidth)
        density_estimates[i] = np.sum(kernels) / (n * variable_bandwidth)

    return density_estimates






def EDR(density, neighbour):
    local_density = density[neighbour]
    local_density = local_density / np.sum(local_density)
    from scipy.stats import entropy
    my_entropy = entropy(local_density)
    EDR = (my_entropy / local_density[0])

    return EDR



#MD Version
def find_k_nearest_neighbors(point, k, tree):
    """
    Find k nearest neighbors of `point` using KD-tree.

    :param point: The point (a list or array) to find the neighbors of.
    :param k: The number of nearest neighbors to find.
    :param tree: KDTree object within which to search.
    :return: Indices of the k-nearest neighbors.
    """
    # Constructing KD-Tree
    distances, indices = tree.query(point, k)
    return indices