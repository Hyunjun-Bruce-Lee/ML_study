# xgboost 설치 : conda install -c anaconda py-xgboost
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Boston housing data set을 읽어온다
boston = load_boston()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x = boston.data
y = boston.target
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# XGBoost (regressor)로 Train 데이터 세트를 학습한다.
model = XGBRegressor(objective='reg:squarederror')  # default로 학습
model.fit(trainX, trainY)

# testX[n]에 해당하는 target (price)을 추정한다.
n = 1
price = model.predict(testX[n].reshape(1,-1))
print('test[%d]의 추정 price = %.2f' % (n, price))
print('test[%d]의 실제 price = %.2f' % (n, testY[n]))
print('추정 오류 = rmse(추정 price - 실제 price) = %.2f' % np.sqrt(np.square(price - testY[n])))

# 시험 데이터 전체의 오류를 MSE로 표시한다.
# MSE는 값의 범위가 크다는 단점이 있다.
predY = model.predict(testX)
rmse = (np.sqrt(mean_squared_error(testY, predY)))
print('\n시험 데이터 전체 오류 (rmse) = %.4f' % rmse)

# 시험 데이터 전체의 오류를 R-square로 표시한다.
print('\n시험 데이터 전체 오류 (R2-score) = %.4f' % model.score(testX, testY))

# R-square를 manual로 계산하고, model.score() 결과와 비교한다.
# SSE : explained sum of square
# SSR : residual sum of square (not explained)
# SST : total sum of square
# R-square : SSE / SST or 1 - (SSR / SST)
ssr = np.sum(np.square(predY - testY))
sst = np.sum(np.square(testY - testY.mean()))
R2 = 1 - ssr / sst
print('R-square = %.4f' % R2)

