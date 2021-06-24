# Logistic Regression으로 iris 데이터를 학습한다.
# multi class classification (class = [0, 1, 2]) &
# Regularization
# ------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x = iris.data
y = iris.target
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# C를 변화시켜가면서 정확도를 측정해 본다
testAcc = []
trainAcc = []
rangeC = np.arange(0.001, 1.0, 0.002)
for C in rangeC:
    # Logistic Regression으로 Train 데이터 세트를 학습한다.
    model = LogisticRegression(penalty='l2', C=C, max_iter=500)
    model.fit(trainX, trainY)
    
    # Test 세트의 Feature에 대한 정확도
    predY = model.predict(testX)
    testAcc.append((testY == predY).mean())
    
    # Train 세트의 Feature에 대한 정확도
    predY = model.predict(trainX)
    trainAcc.append((trainY == predY).mean())

plt.figure(figsize=(8, 5))
plt.plot(rangeC, testAcc, label="Test Data")
plt.plot(rangeC, trainAcc, label="Train Data")
plt.legend()
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.show()


