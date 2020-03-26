
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
dataset = pd.read_csv('studentinfoCHK.csv')
X = dataset.iloc[10000:20000, [6, 7]].values
mb_clustering = clustering = MiniBatchKMeans(n_clusters=2)
y_mb_clustering= mb_clustering.fit_predict(X)
plt.scatter(X[y_mb_clustering == 0, 0], X[y_mb_clustering == 0, 1], s = 20, c = 'red', label = 'high score and high study credits')
plt.scatter(X[y_mb_clustering == 1, 0], X[y_mb_clustering == 1, 1], s = 20, c = 'blue', label = 'high score and low study credits')
plt.title('Clusters of STUDENTS')
plt.xlabel('SCORE')
plt.ylabel('STUDIED CREDITS')
plt.plot()
