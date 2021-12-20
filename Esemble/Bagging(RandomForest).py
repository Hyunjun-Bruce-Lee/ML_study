# ------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# iris 데이터를 읽어온다.
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(iris['data'], iris['target'], test_size = 0.2)

rf = RandomForestClassifier(max_depth=5, n_estimators=100)
rf.fit(trainX, trainY)

# 학습데이터와 시험데이터의 정확도를 측정한다.
print('\n학습데이터의 정확도 = %.2f' % rf.score(trainX, trainY))
print('시험데이터의 정확도 = %.2f' % rf.score(testX, testY))

# 시험데이터의 confusion matrix를 작성한다 (row : actual, col : predict)
predY = rf.predict(testX)
print('\nConfusion matrix :')
print(confusion_matrix(testY, predY))
print()
print(classification_report(testY, predY, target_names=iris.target_names))

# Sub tree별 시험데이터의 정확도를 확인한다.
print('\nSubtree별 시험데이터 정확도 :')
for i in range(5):
    subTree = rf.estimators_[i]
    print('subtree (%d) = %.2f' % (i, subTree.score(testX, testY)))

# classification_report()를 해석해 본다.
import numpy as np
label = np.vstack([testY, predY]).T

# class = n이라고 예측한 것 중 실제 classe=n인 비율
def precision(n):
    y = label[label[:, 1] == n]
    match = y[y[:, 0] == y[:, 1]]
    return match.shape[0] / y.shape[0]

print()
print('class-0 precision : %.2f' % precision(0))
print('class-1 precision : %.2f' % precision(1))
print('class-2 precision : %.2f' % precision(2))

# 실제 class = n인 것중 classe=n으로 예측한 비율
def recall(n):
    y = label[label[:, 0] == n]
    match = y[y[:, 0] == y[:, 1]]
    return match.shape[0] / y.shape[0]

print()
print('class-0 recall : %.2f' % recall(0))
print('class-1 recall : %.2f' % recall(1))
print('class-2 recall : %.2f' % recall(2))

# F1-score
def f1_score(n):
    p = precision(n)
    r = recall(n)
    return 2 * p * r / (p + r)

print()
print('class-0 f1-score : %.2f' % f1_score(0))
print('class-1 f1-score : %.2f' % f1_score(1))
print('class-2 f1-score : %.2f' % f1_score(2))
