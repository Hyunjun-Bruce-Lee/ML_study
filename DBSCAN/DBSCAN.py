from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)

db = DBSCAN(eps = 0.2, min_samples = 5, metric = 'euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c = 'R', marker = 'o', s= 40, edgecolor = 'black')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c = 'Y', marker = 's', s= 40, edgecolor = 'black')
plt.legend()
plt.tight_layout()


###############################################################
import pandas as pd
import numpy as np


with open('./dataset/mnist.pickle', 'rb') as f:
        mnist = pickle.load(f)

mnist_df = pd.DataFrame(mnist.data, columns = mnist.feature_names)
target = pd.DataFrame(mnist.target, columns = ['target'])
mnist_df = mnist_df.iloc[0:2000,:]
target = target.iloc[0:2000,:]

from sklearn.manifold import TSNE

model = TSNE(learning_rate=300)
transformed = model.fit_transform(mnist_df)

db = DBSCAN(eps = 2.4 , min_samples = 10, metric = 'euclidean')
y_db = db.fit_predict(transformed)
y_db

predict = db.fit(transformed)
y_pred = predict.labels_

dataset = pd.DataFrame({'Column1':transformed[:,0],'Column2':transformed[:,1]})
dataset['cluster_num'] = pd.Series(predict.labels_)

imageX = np.array(mnist_df).copy()
# cluster 별로 이미지를 확인한다.
f = plt.figure(figsize=(8, 2))
for k in np.unique(clust):
    # cluster가 i인 imageX image 10개를 찾는다.
    idx = np.where(clust == k)[0][:10]
    
    f = plt.figure(figsize=(8, 2))
    for i in range(10):
        image = imageX[idx[i]].reshape(28,28)
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(image, cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(k)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()

############################################33

inputX = mnist.data[:2000, :]
imageX = inputX.copy()

db = DBSCAN(eps = 3.77 , min_samples = 10, metric = 'euclidean')
clust = db.fit_predict(transformed)
#clust = db.fit_predict(inputX)


f = plt.figure(figsize=(8, 2))
for k in np.unique(clust):
    idx = np.where(clust == k)[0][:10]
    
    f = plt.figure(figsize=(8, 2))
    for i in range(10):
        image = imageX[idx[i]].reshape(28,28)
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(image, cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(k)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
