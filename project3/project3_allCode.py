import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE

"""
TASK 1
"""

# reading the data into a data frame
df = pd.read_csv("Swiss_Roll.txt", sep='\s+', header=None)
df.columns = ["y1", "y2", "y3"]

# 3d plot of the swiss roll, colored based on z-value ("y3")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
coloring = df["y3"]
ax.scatter(df["y1"], df["y2"], df["y3"], c=coloring)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.figure.savefig("swissRoll", dpi=600)

lle = LLE(n_components=2, n_neighbors=35).fit(df)
embedding_lle = lle.embedding_
reconstruction_error_lle = lle.reconstruction_error_

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(embedding_lle[:,0], embedding_lle[:,1], c=coloring)
ax.set_xticks([])
ax.set_yticks([])
ax.figure.savefig("higherK", dpi=600)

"""
TASK 2
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Arbitrary_Shape.txt", sep='\s+', header=None)
df.columns = ["y1", "y2"]
df = np.array(df)
df = StandardScaler().fit_transform(df)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(df[:,0], df[:,1], s=0.4)
ax.figure.savefig("arbitraryShape", dpi=600)

# e_incrs = np.linspace(0.05, 0.15, num=15)
# minPoints_incrs = [i for i in range(4,20)]

# silh_scores = []
# for e in e_incrs:
#     for mP in minPoints_incrs:
#         db = DBSCAN(eps=e, min_samples=mP).fit(df)
#         labels = db.labels_
#         silh_scores.append([e, mP, silhouette_score(df, labels)])

# silh_scores = np.array(silh_scores)
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(silh_scores[:,0], silh_scores[:,1], s= 0.4)

db = DBSCAN(eps=0.11, min_samples=14).fit(df)
labels = db.labels_

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = df[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=4,
    )

    xy = df[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=2,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.savefig("higherEpsilon.png", dpi=600)
plt.show()

# removing the noise from the data
idx = np.where(labels != -1)[0]
labels_wo_noise = labels[idx]
df_wo_noise = df[idx, :]

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(df_wo_noise[:,0], df_wo_noise[:,1], s=0.4)
ax.figure.savefig("arbitraryShapeNoNoise", dpi=600)

silhouette_avg = silhouette_score(df_wo_noise, labels_wo_noise)
each_silhouette_score = silhouette_samples(df_wo_noise, labels_wo_noise)

colorlist =["tomato","mediumseagreen","blueviolet","cornflowerblue",
            "darkgreen","skyblue"]

#Visualization
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
y_lower =10
for i in range(n_clusters_):
    ith_cluster_silhouette_values = each_silhouette_score[labels_wo_noise == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = colorlist[i]
    ax.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.3)
    
    #label the silhouse plots with their cluster numbers at the middle
    ax.text(-0.05,y_lower + 0.5 * size_cluster_i,str(i))
    
    #compute the new y_lower for next plot
    y_lower = y_upper +10 
    
ax.set_title("Silhuoette plot")
ax.set_xlabel("silhouette score")
ax.set_ylabel("Cluster label")
    
#the vertical line for average silhouette score of all the values
ax.axvline(x=silhouette_avg,color="red",linestyle="--")
    
ax.set_yticks([])
ax.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1])
ax.figure.savefig("silhouettePlot", dpi=600)
