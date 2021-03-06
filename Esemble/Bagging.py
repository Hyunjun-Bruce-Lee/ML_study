from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
import numpy as np

# iris 데이터를 읽어온다.
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(iris['data'], iris['target'], test_size = 0.2)

# 4가지 모델을 생성한다 (KNN, Decision tree, SVM, Logistic Regression).
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
dtree = DecisionTreeClassifier(criterion='gini', max_depth=8)
svm = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
lreg = LogisticRegression(max_iter=500)

# 4가지 모델로 Bagging을 구성하고, 각 모델의 추정 확률을 누적한다.
prob = np.zeros((testY.shape[0], iris.target_names.shape[0]))   
base_model = [knn, dtree, svm, lreg]

validation_dict = dict()
for m in base_model:
    bag = BaggingClassifier(base_estimator=m, n_estimators=100, bootstrap=True)
    bag.fit(trainX, trainY)
    temp_proba = bag.predict_proba(testX)
    validation_dict[str(m)] = temp_proba
    prob += temp_proba

# 확률의 누적합이 가장 큰 class를 찾고, 정확도를 측정한다.
predY = np.argmax(prob, axis=1)
accuracy = (testY == predY).mean()
for i in validation_dict.keys():
    temp_pred = np.argmax(validation_dict[i], axis = 1)
    temp_acc = (testY == temp_pred).mean()
    print(f'{i} : {temp_acc}')
print(f'Esemble : {accuracy}')

# 시험데이터의 confusion matrix를 작성한다 (row : actual, col : predict)
print('\nConfusion matrix :')
print(confusion_matrix(testY, predY))









wine = load_wine()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(wine['data'], wine['target'], test_size = 0.2)

# 4가지 모델을 생성한다 (KNN, Decision tree, SVM, Logistic Regression).
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
dtree = DecisionTreeClassifier(criterion='gini', max_depth=8)
svm = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
lreg = LogisticRegression(max_iter=10000)

# 4가지 모델로 Bagging을 구성하고, 각 모델의 추정 확률을 누적한다.
prob = np.zeros((testY.shape[0], iris.target_names.shape[0]))   
base_model = [knn, dtree, svm, lreg]

validation_dict = dict()
for m in base_model:
    bag = BaggingClassifier(base_estimator=m, n_estimators=100, bootstrap=True)
    bag.fit(trainX, trainY)
    temp_proba = bag.predict_proba(testX)
    validation_dict[str(m)] = temp_proba
    prob += temp_proba

# 확률의 누적합이 가장 큰 class를 찾고, 정확도를 측정한다.
predY = np.argmax(prob, axis=1)
accuracy = (testY == predY).mean()
for i in validation_dict.keys():
    temp_pred = np.argmax(validation_dict[i], axis = 1)
    temp_acc = (testY == temp_pred).mean()
    print(f'{i} : {temp_acc}')
print(f'Esemble : {accuracy}')

# 시험데이터의 confusion matrix를 작성한다 (row : actual, col : predict)
print('\nConfusion matrix :')
print(confusion_matrix(testY, predY))
