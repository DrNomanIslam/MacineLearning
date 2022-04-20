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

print("Finding the appropriate number of clusters")

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    meandistortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis = 1)) / data.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()