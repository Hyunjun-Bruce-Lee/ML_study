# Decision Tree로 cancer 데이터를 학습한다.
# ----------------------------------------
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = \
    train_test_split(cancer['data'], cancer['target'], test_size = 0.2)

trainGini = []
testGini = []
trainEntropy = []
testEntropy = []
for k in range(1, 15):
    # Gini 계수를 사용하여 학습 데이터를 학습한다.
    dt = DecisionTreeClassifier(criterion='gini', max_depth=k)
    dt.fit(trainX, trainY)
    
    # 정확도를 측정한다.
    trainGini.append(dt.score(trainX, trainY))
    testGini.append(dt.score(testX, testY))
    
    # Entropy를 사용하여 학습 데이터를 학습한다.
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=k)
    dt.fit(trainX, trainY)
    
    # 정확도를 측정한다.
    trainEntropy.append(dt.score(trainX, trainY))
    testEntropy.append(dt.score(testX, testY))

# Gini와 Entropy, 그리고 tree depth에 따른 정확도를 비교한다.
plt.figure(figsize=(8, 5))
plt.plot(trainGini, label="Gini/Train")
plt.plot(trainEntropy, label="Entropy/Train")
plt.plot(testGini, label="Gini/Test")
plt.plot(testEntropy, label="Entropy/Test")
plt.legend()
plt.xlabel("Tree depth")
plt.ylabel("Accuracy")
plt.show()


