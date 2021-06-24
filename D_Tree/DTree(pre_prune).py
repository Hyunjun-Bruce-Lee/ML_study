# Decision Tree로 income 데이터를 학습한다 (pre-pruning).
# 데이터 세트 : http://archive.ics.uci.edu/ml/datasets/Adult
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 데이터 파일을 읽어온다.
income = pd.read_csv("dataset/income.csv", index_col=False)

# categorical feature들을 숫자로 바꾼다.
cat_features = ["workclass", "education_num", "marital_status", 
                "occupation", "relationship", "race", 
                "sex","native_country", "income"]
for c in cat_features:
    income[c] = pd.Categorical(income[c]).codes
    
data = np.array(income)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x = data[:, 0:10]
y = data[:, -1]
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

trainGini = []
testGini = []
trainEntropy = []
testEntropy = []
depth = []
for k in range(1, 20):
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
    
    depth.append(k)
    print('depth = %d done.' % k)

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

# 정확도가 가장 큰 최적 depth를 찾는다.
nDepth = depth[np.argmax(testGini)]

# opt_alpha를 적용한 tree를 사용한다.
dt = DecisionTreeClassifier(max_depth = nDepth)
dt.fit(trainX, trainY)
print('시험 데이터의 정확도 = %.4f' % dt.score(testX, testY))
print('최적 트리의 depth = %d' % nDepth)

