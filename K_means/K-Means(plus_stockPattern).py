# K-Means++ Clustering 연습
# 주가 패턴 분류
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# stock pattern 데이터 세트를 읽어온다
ds = pd.read_csv('dataset/stockPattern.csv')
X = np.array(ds)
K = 8

# K-means++ 알고리즘으로 학습 데이터를 K 그룹으로 분류한다.
km = KMeans(n_clusters=K, init='k-means++', n_init=3, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(X)
y_km = km.predict(X)

# Centroid pattern을 그린다
fig = plt.figure(figsize=(10, 6))
colors = "bgrcmykw"
centXY = km.cluster_centers_
for i in range(K):
    s = 'pattern-' + str(i)
    p = fig.add_subplot(2, (K+1)//2, i+1)
    p.plot(centXY[i], 'b-o', markersize=3, color=colors[np.random.randint(0, 7)], linewidth=1.0)
    p.set_title('Cluster-' + str(i))

plt.tight_layout()
plt.show()

# 데이터 패턴 몇 개만 그려본다.
cluster = 0
ds['cluster'] = y_km
plt.figure(figsize=(6, 6))
p = ds.loc[ds['cluster'] == cluster]
p = p.sample(frac=1).reset_index(drop=True)
for i in range(10):
    plt.plot(p.iloc[i][0:20])
    
plt.title('Cluster-' + str(cluster))
plt.show()
