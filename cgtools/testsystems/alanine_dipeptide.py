import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import pyemma
from msmbuilder.example_datasets import AlanineDipeptide
import matplotlib.pyplot as plt

# fetch trajectories
trajs = AlanineDipeptide().get().trajectories

# featurize using raw phi and psi backbone torsion angles
feat = pyemma.coordinates.featurizer(trajs[0].top)
feat.add_backbone_torsions()
X = [feat.transform(traj) for traj in trajs]

# cluster into way too many microstates
kmeans = pyemma.coordinates.cluster_mini_batch_kmeans(X, k=500, max_iter=1000)
dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]

# estimate markov model
msm = pyemma.msm.estimate_markov_model(dtrajs, lag=10)

# voronoi plot
X_ = np.vstack(X)
inds = np.hstack(dtrajs)
# c = inds # color by cluster ID
c = np.array([msm.stationary_distribution[i] for i in inds]) # color by stationary distribution
# c = np.log([msm.stationary_distribution[i] for i in inds]) # color by free energy


def create_plot(centers, points, colors, cmap='Blues'):
    voronoi = Voronoi(centers)
    plot = voronoi_plot_2d(voronoi, show_points=False, show_vertices=False, line_width=0.5)
    plt.scatter(points[:,0], points[:,1], linewidths=0, c=colors, s=1, cmap=cmap)

    # axes and labels
    plt.xlabel(feat.describe()[0])
    plt.ylabel(feat.describe()[1])
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.title('Alanine dipeptide discretized by backbone dihedrals')

create_plot(kmeans.clustercenters, X_, c)

# apply coarse-graining and plot again...