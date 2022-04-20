from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy.spatial.distance import cdist

df = pd.read_csv('data.csv')
df1 = df.drop(df.columns[[0, 1]], axis=1)
data = df1.as_matrix()
K = range(1, 10)
print("Finding the appropriate number of clusters")
K = range(2, 10)
maxK=-1
max=-1
scs=[]
km= KMeans(n_clusters=2)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    sc=metrics.silhouette_score(data, kmeans.labels_, metric='euclidean')
    scs.append(sc)
    if maxK==-1 or sc>max:
        maxK=k
        km=kmeans
        max=sc

plt.plot(K, scs, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Silhouette coefficient Method')
plt.show()
