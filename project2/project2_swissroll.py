import pandas as pd
import scipy as sp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Swiss Roll Task 2
"""

df = pd.read_csv("Swiss_Roll.txt", sep='\s+', header=None)
df.columns = ["y1", "y2", "y3"]

ax = plt.axes(projection = '3d')
ax.scatter(df["y1"], df["y2"], df["y3"])
plt.show()

dist_matrix = sp.spatial.distance_matrix(df, df)

nbrs = NearestNeighbors(n_neighbors=5,
                        metric="precomputed", 
                        algorithm="brute").fit(dist_matrix)

W = nbrs.kneighbors_graph(dist_matrix).toarray()

I = np.identity(2000)
W = W - I
W = W + np.transpose(W)
W[W > 1] = 1

nbr_dists = W * dist_matrix
sigma = np.mean(nbr_dists[nbr_dists > 0])
temp_weights = np.exp(- nbr_dists**2 / (2*sigma**2))

W = temp_weights * W
D = np.diag([sum(row) for row in W])
L = D - W

eigvals, eigvecs = np.linalg.eigh(L)

volV = np.sum(D)

L_pinv = sp.linalg.pinv(L)
CTD_un = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        CTD_un[i,j] = volV*(L_pinv[i,i] - 2*L_pinv[i,j] + L_pinv[j,j])
    
CTD_un = np.array(CTD_un)
eigvals_un, eigvecs_un = np.linalg.eigh(CTD_un)
plt.plot(eigvals_un, ".")

embedding_un = MDS(n_components = 5, dissimilarity = "precomputed")
MDS_un = embedding_un.fit_transform(CTD_un)


plt.plot(MDS_un[:,0], MDS_un[:,1], ".")

"""
Now to the normalized Laplacian
"""

D_invsqrt = np.diag(1 / np.sqrt([sum(row) for row in W]))
L_sym = np.matmul(D_invsqrt, np.matmul(L, D_invsqrt))
eigvals_sym, eigvecs_sym = np.linalg.eigh(L_sym)

# based on the Laplacian L_sym
CTD_n = np.zeros((len(df), len(df)))
L_sym_pinv = sp.linalg.pinv(L_sym)
for i in range(len(df)):
    for j in range(len(df)):
        CTD_n[i,j] = volV*(L_sym_pinv[i,i]- 2*L_sym_pinv[i,j] +L_sym_pinv[j,j])
    
CTD_n = np.array(CTD_n)
eigvals_n, eigvecs_n = np.linalg.eigh(CTD_n)
plt.plot(eigvals_n, ".")

embedding_n = MDS(n_components = 5, dissimilarity = "precomputed")
MDS_n = embedding_n.fit_transform(CTD_n)

plt.scatter(MDS_n[:,0], MDS_n[:,1], s = 0.5)
