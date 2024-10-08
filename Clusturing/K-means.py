import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans
#charger les données
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60,
random_state=0)
#affichage initial
plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()
#application de l'apprentissage KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
#tester la prédiction
y_kmeans = kmeans.predict(X)
#Affichage graphique des centres de clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9);
plt.show()