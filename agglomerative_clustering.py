
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
# Importing the dataset
dataset = pd.read_csv('studentinfoCHK.csv')
X = dataset.iloc[10000:20000, [6, 7]].values


s_clustering =  AgglomerativeClustering(n_clusters=2,affinity='euclidean', linkage = 'complete')
y_s_clustering= s_clustering.fit_predict(X)
# Visualising the clusters

plt.scatter(X[y_s_clustering == 0, 0], X[y_s_clustering == 0, 1], s = 20, c = 'red', label = 'high score and high study credits')
plt.scatter(X[y_s_clustering == 1, 0], X[y_s_clustering == 1, 1], s = 20, c = 'blue', label = 'high score and low study credits')
plt.title('Clusters of STUDENTS')
plt.xlabel('SCORE')
plt.ylabel('STUDIED CREDITS')
plt.legend()
plt.plot()
plt.savefig('Graphs/AgglomerativeClustering.png')