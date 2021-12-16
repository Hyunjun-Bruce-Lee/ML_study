from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

# iris 데이터를 가져온다.
iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(iris['data'], iris['target'], test_size = 0.2)

# XGBoost (classifier)로 Train 데이터를 학습한다.
# 학습데이터와 시험데이터를 xgb의 데이터 형태로 변환한다.
trainD = xgb.DMatrix(trainX, label = trainY)
testD = xgb.DMatrix(testX, label = testY)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',
    # multi:softmax –set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
    # multi:softprob –same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
    'num_class': 3}   # class 개수 = 3개 (multi class classification)

model = xgb.train(params = param, dtrain = trainD, num_boost_round = 20)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = model.predict(testD)
predY = np.argmax(predY, axis=1)
accuracy = (testY == predY).mean()
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

# Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = model.predict(trainD)
predY = np.argmax(predY, axis=1)
accuracy = (trainY == predY).mean()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)
