from scipy.cluster import hierarchy
from utils import total_variation, cg_T
import matplotlib.pyplot as plt
import numpy as np

def hierarchy_experiment(T, pi, tau, n_macro = range(2, 99)):
    after_propagation = np.linalg.matrix_power(T, tau)

    Zs = dict()
    linkage_types = ['single', 'complete', 'average', 'weighted',
                     #'centroid', 'median', 'ward' # these require distance metric to be Euclidean
                    ]
    for linkage in linkage_types:
        Zs[linkage] = hierarchy.linkage(after_propagation, method = linkage, metric=total_variation)

    for linkage in linkage_types:
        metastabs = map(lambda i: np.trace(
            cg_T(T, pi, hierarchy.cut_tree(Zs[linkage], i).flatten())), n_macro)
        plt.plot(n_macro, metastabs, '.', label=linkage)
    plt.legend(loc='best')
    plt.xlabel('# macrostates')
    plt.ylabel('Trace of macrostate transition matrix')


    plt.figure()
    for linkage in linkage_types:
        metastabs = map(lambda i: np.trace(cg_T(T, pi, hierarchy.cut_tree(Zs[linkage], i).flatten())), n_macro)
        plt.plot(n_macro, np.array(metastabs) / np.array(n_macro), '.', label=linkage)
    plt.legend(loc='best')
    plt.xlabel('# macrostates')
    plt.ylabel('"Fractional metastability" of macrostate transition matrix')