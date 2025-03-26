import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Arbitrary_Shape.txt", sep='\s+', header=None)
df.columns = ["y1", "y2"]

plt.scatter(df["y1"], df["y2"], s=0.3)