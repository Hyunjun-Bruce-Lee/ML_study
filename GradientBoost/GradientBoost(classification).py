# GBM for classification
# -------------------------------------------------------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# breast cancer 데이터를 가져온다.
cancer = load_breast_cancer()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(cancer['data'], cancer['target'], test_size = 0.2)

# Gradient Boosting (classifier)로 Train 데이터 세트를 학습한다.
# default:
# loss = deviance (=logistic regression)
# learning_rate = 0.1
# n_estimators = 100
# max_depth = 3
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                 n_estimators=100, max_depth=3)
model.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
# accuracy = model.score(testX, testY)와 동일함.
predY = model.predict(testX)
accuracy = (testY == predY).mean()
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

# Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = model.predict(trainX)
accuracy = (trainY == predY).mean()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)
