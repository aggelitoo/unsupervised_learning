import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE


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
ax.figure.savefig("lleEmbedding", dpi=600)

