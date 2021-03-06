# linear-SVM으로 iris 데이터를 학습한다.
# -------------------------------------
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mglearn

iris = load_iris()

# 시각화를 위해 sepal length와 sepal width 만 사용한다.
x = iris.data[:, [0, 1]] # colume 0과 1만 사용함.
y = iris.target

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# SVM으로 Train 데이터 세트를 학습한다.
model = SVC(kernel='linear', C = 1)
model.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % model.score(trainX, trainY))
print("* 시험용 데이터로 측정한 정확도 = %.2f" % model.score(testX, testY))

# 시각화
plt.figure(figsize=[9,7])
mglearn.plots.plot_2d_classification(model, trainX, alpha=0.1)
mglearn.discrete_scatter(trainX[:,0], trainX[:,1], trainY)
plt.legend(iris.target_names)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

