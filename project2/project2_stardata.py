import pandas as pd
import scipy as sp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

"""
Star data task
"""

# a 300x2 dataframe (matrix) containing "star data"
df = pd.read_csv("Star_Data.txt", sep='\s+', header=None)
df.columns = ["y1", "y2"]

"""
Can actually see from this plot that k=5 should be enough to get a fully
connected graph using nearest neighbor (undirected, i.e. made symmetric).
That this k will suffice can be seen from the right-most point in the following
plot.
"""
plt.plot(df["y1"], df["y2"], "y*")
plt.title('In')
plt.show()

"""
Assuming Euclidean distances are appropriate for the Star_Data.txt file, we
create the dissimilarity matrix (distance matrix) using the 
"""
dist_matrix = sp.spatial.distance_matrix(df, df)


"""
Using the sklearn nearest neighbor function in order to create an unweighted
adjacency matrix. Since this function creates self-edges, we have to choose
k+1 as the input to the function.
"""
nbrs = NearestNeighbors(n_neighbors=6,
                        metric="precomputed", 
                        algorithm="brute").fit(dist_matrix)

# getting the unweighted adjacency matrix
W = nbrs.kneighbors_graph(dist_matrix).toarray()

# removing self-edges
I = np.identity(300)
W = W - I

# making graph undirected (symmetric)
W = W + np.transpose(W)
W[W > 1] = 1


"""
Weighing the edges using a t-dist. 
Setting the hyperparam. dof = v to the mean distance such that it is "favored"
that the squared distance and sigma^2 cancel eachother out.
"""
nbr_dists = W * dist_matrix
v = 1
temp_weights = (1 + dist_matrix**2 / v)**(-(v + 1) / 2)

# putting the weights into the adjacency matrix W
W = temp_weights * W

# retrieving the diagonal degrees matrix
D = np.diag([sum(row) for row in W])

# unnormalized Laplacian
L = D - W

"""
Approximating eigenvalues and sorting in ascending order. If there is only one
zero-eigenvalue, then the graph is fully connected (according to proposition
in von Luxburg).
"""

# eigvals and vecs from unnormalized Laplacian
eigvals, eigvecs = np.linalg.eigh(L)

# two clusters => need first two eigenvectors
V = eigvecs[0:5].T

# plotting the eigvals in order to deduce if there are any spectral gaps
plt.plot(eigvals, ".", label = "eigenvalues")
plt.show()

plt.plot(V[:,0], V[:,1], ".")
plt.show()

"""
Now to the normalized Laplacian
"""
D_invsqrt = np.diag(1 / np.sqrt([sum(row) for row in W]))
L_sym = np.matmul(D_invsqrt, np.matmul(L, D_invsqrt))

eigvals_sym, eigvecs_sym = np.linalg.eigh(L_sym)

V_sym = eigvecs_sym[0:2].T
U = []

for i in range(len(V_sym)):
    temp = []
    for j in range(len(V_sym[0])):
        temp.append(V_sym[i, j] / np.sqrt(sum(V_sym[i]**2)))
    U.append(temp)

U = np.array(U)

plt.plot(eigvals_sym, ".")
plt.show()

plt.plot(U[:,0], U[:,1], ".")
plt.show()

plt.plot(eigvecs[0], ".")


