from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# iris 데이터를 읽어온다.
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(iris['data'], iris['target'], test_size = 0.2)

# 4가지 모델을 생성한다 (KNN, Decision tree, SVM, Logistic Regression).
# 각 모델은 최적 조건으로 생성한다. (knn의 k개수, dtree의 max_depth 등)
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
dtree = DecisionTreeClassifier(criterion='gini', max_depth=8)
svm = SVC(kernel='rbf', gamma=0.1, C=1.0, probability=True)
lreg = LogisticRegression(max_iter=500)

# 4가지 모델로 앙상블을 구성한다.
base_model = [('knn', knn), ('dtree', dtree), ('svm', svm), ('lr', lreg)]
ensemble = VotingClassifier(estimators=base_model, voting='soft')

# 4가지 모델을 각각 학습하고, 결과를 종합한다.
ensemble.fit(trainX, trainY)

# 학습데이터와 시험데이터의 정확도를 측정한다.
print('\n학습데이터의 정확도 = %.2f' % ensemble.score(trainX, trainY))
print('시험데이터의 정확도 = %.2f' % ensemble.score(testX, testY))

# 시험데이터의 confusion matrix를 작성한다 (row : actual, col : predict)
predY = ensemble.predict(testX)
print('\nConfusion matrix :')
print(confusion_matrix(testY, predY))
