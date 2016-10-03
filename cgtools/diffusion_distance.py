import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import SpectralClustering

def diffusion_distance_cg(T, tau=50, threshold=1.5, max_levels=10):
    '''

    Parameters
    ----------
    T : numpy.ndarray, (n, n)
        microstate transition matrix to coarse-grain
    tau : integer
        timescale of interest, as an integer multiple of the transition matrix lag_time
    threshold : number > 1
        threshold used to choose the number of macrostates
    max_levels : positive integer
        maximum number of choices of n_macrostates to consider

    Returns
    -------
    cg_maps : list of integer arrays
        each element of cg_maps is a length-n integer array, mapping
        each microstate in range(n) to an integer label in range(n_macrostates)
    '''


    # compute the tau-step transition matrix from tau
    after_propagation = np.linalg.matrix_power(T, tau)

    # compute an affinity matrix using these tau-step transition probabilities
    affinity_matrix = 1 - 0.5 * squareform(pdist(after_propagation, p=1))

    # select the number of macrostates, by inspecting the spectrum of the affinity matrix
    # we'll choose to "cut" at n_macrostates where there's a "spectral gap" of a factor greater than `threshold`
    vals, vecs = np.linalg.eigh(affinity_matrix)
    vals_ = vals[::-1]
    ks = np.where((vals_[:-1] / vals_[1:]) > threshold)[0] + 2
    ks = ks[ks < len(T)]

    # if this procedure identifies too many levels, just take the top-max_levels of them
    if len(ks) > max_levels:
        print('yikes! too many levels!')
        ks = ks[:max_levels]

    # retrieve coarse-graining maps by spectral clustering of the affinity matrix
    cg_maps = []
    for k in ks:
        clust = SpectralClustering(n_clusters=k, affinity='precomputed')
        clust.fit(affinity_matrix)
        cg_maps.append(clust.labels_)

    return cg_maps