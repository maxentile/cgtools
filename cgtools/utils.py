'''
Random utility functions I don't know where to put yet..
'''

from numba import jit
import numpy as np
import pyemma
from sklearn.cluster import SpectralClustering

def total_variation(p, q):
    '''
    Computes the total variation distance between distributions p and q
    (assumes p and q are discrete distributions: normalized)
    '''
    return 0.5 * np.linalg.norm(p - q, 1)

def crisp_assign_using_hmm(i, hmm):
    return np.argmax(hmm.observation_probabilities[:,i])

@jit
def cg_T(microstate_T, microstate_pi, cg_map):
    '''
    Coarse-grain a microstate transition matrix by applying cg_map

    Parameters
    ----------
    microstate_T : (N,N), array-like, square
        microstate transition matrix
    microstate_pi : (N,), array-like
        microstate stationary distribution
    cg_map : (N,), array-like
        assigns each microstate i to a macrostate cg_map[i]

    Returns
    -------
    T : numpy.ndarray, square
        macrostate transition matrix
    '''

    n_macrostates = np.max(cg_map) + 1
    n_microstates = len(microstate_T)

    # compute macrostate stationary distribution
    macrostate_pi = np.zeros(n_macrostates)
    for i in range(n_microstates):
        macrostate_pi[cg_map[i]] += microstate_pi[i]
    macrostate_pi /= np.sum(macrostate_pi)

    # accumulate macrostate transition matrix
    T = np.zeros((n_macrostates, n_macrostates))
    for i in range(n_microstates):
        for j in range(n_microstates):
            T[cg_map[i], cg_map[j]] += microstate_pi[i] * microstate_T[i, j]

    # normalize
    for a in range(n_macrostates):
        T[a] /= macrostate_pi[a]

    return T


def create_merged_assignment_vector(cg_map, pair_to_merge):
    '''

    Parameters
    ----------

    cg_map : length-N vector, with entries in range(m)

    pair_to_merge : i,j in range(m)

    Returns
    -------

    merged_cg_map : length-N vector, with entries in range(m-1)

    '''
    merged_cg_map = np.array(cg_map)

    # later assume j > i
    i, j = sorted(pair_to_merge)

    # assign anything from cluster j to cluster i
    merged_cg_map[merged_cg_map == j] = i

    # fill in indices, if necessary
    merged_cg_map[merged_cg_map == np.max(cg_map)] = j

    return merged_cg_map

def score_merge(T, pi, assignment, pair_to_merge):
    merged_cg_map = create_merged_assignment_vector(assignment, pair_to_merge)
    new_T = cg_T(T, pi, merged_cg_map)
    return np.trace(new_T)


def hierarchical_T_clustering(T, pi):
    cg_map = np.arange(len(T))
    new_cg_map = np.arange(len(T))

    metastabilities = [np.trace(T)]
    merge_matrices = []
    all_cg_maps = [cg_map]

    for t in range(1, len(T) - 1):
        best = - np.inf
        n_macro = np.max(cg_map) + 1
        merge_matrix = np.zeros((n_macro, n_macro))

        for i in range(n_macro):
            for j in range(i):
                merged_cg_map = create_merged_assignment_vector(cg_map, (i, j))
                new_T = cg_T(T, pi, merged_cg_map)
                trace = np.trace(new_T)
                merge_matrix[i, j] = trace

                if trace > best:
                    best = trace
                    new_cg_map = merged_cg_map

        merge_matrices.append(merge_matrix + merge_matrix.T)
        cg_map = new_cg_map
        all_cg_maps.append(new_cg_map)
        metastabilities.append(best)

    return merge_matrices, metastabilities, all_cg_maps

def compute_mean_exit_times(msm):
    mean_exit_times = np.zeros(msm.nstates)

    for i in range(msm.nstates):
        not_i = list(set(range(msm.nstates)) - set([i]))
        mean_exit_times[i] = msm.mfpt(i, not_i)
    return mean_exit_times

def get_min_mean_exit_time_cg(diffusion_distance_matrix, dtrajs, n_states):
    clust = SpectralClustering(n_clusters=n_states, affinity='precomputed')
    clust.fit(1.0/(diffusion_distance_matrix+0.1))
    cg_dtrajs = [np.array([clust.labels_[i] for i in dtraj]) for dtraj in dtrajs]
    cg_msm = pyemma.msm.estimate_markov_model(cg_dtrajs, 10)
    mean_exit_times = compute_mean_exit_times(cg_msm)
    return min(mean_exit_times)