# Logistic Regression으로 credit 데이터를 학습한다.
# binary classification (class = [0, 1])
# ------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Credit data set을 읽어온다
credit = pd.read_csv('./dataset/credit_data.csv')
credit = np.array(credit, dtype=np.float32)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x = credit[:, 0:6]
y = credit[:, -1]
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# Logistic Regression으로 Train 데이터 세트를 학습한다.
model = LogisticRegression()
model.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % model.score(trainX, trainY))
print("* 시험용 데이터로 측정한 정확도 = %.2f" % model.score(testX, testY))

# 학습된 w, b를 확인해 본다.
print('\nw :')
print(model.coef_)
print('\nb :')
print(model.intercept_)
print('\nclass :')
print(model.classes_)

# textX[0]의 class를 추정한다.
model.predict(testX)[0]
print('\ntextX[0]의 class :')
print('prob = ', model.predict_proba(testX)[0])

# manual로 testX[0]의 class를 추정해 본 ㅔㅠ[];ㅛ'다. 각 파라메터의 기능을 확인한다.
theta = np.dot(model.coef_[0], testX[0]) + model.intercept_
prob = 1.0 / (1.0 + np.exp(-theta))
print('prob = ', prob)
