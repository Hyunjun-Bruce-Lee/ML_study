# Naive Bayes로 credit 데이터를 학습한다.
# feature들이 모두 실숫값이므로 gaussian model을 사용한다.
# ------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# iris data set을 읽어온다
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(iris.data, iris.target, test_size = 0.2)

# Gaussian model로 Train 데이터 세트를 학습한다.
# GaussianNB() : feature전체가 실수형일때 사용. 전체data를 정규분포화 한다.
modelG = GaussianNB()
modelG.fit(trainX, trainY)

print("\n* Gaussian model :")
print("* 학습용 데이터로 측정한 정확도 = %.2f" % modelG.score(trainX, trainY))
print("* 시험용 데이터로 측정한 정확도 = %.2f" % modelG.score(testX, testY))

# number of training samples observed in each class.
modelG.class_count_

# 평균, 표준편차
modelG.theta_
modelG.sigma_
