# Naive Bayes 분류기로 income 데이터 세트를 학습한다.
# categorical과 gaussian feature가 섞여 있는 경우, 각 feature를 분리하여 MultinomialNB와
# GaussianNB로 나눠서 학습하고 추정 확률을 곱한 값으로 시험 데이터의 label을 추정한다.
# 데이터 세트 : http://archive.ics.uci.edu/ml/datasets/Adult
# ------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# 데이터 파일을 읽어온다.
income = pd.read_csv("dataset/income.csv", index_col=False)

# categorical feature를 숫자로 변환한다.
cat_features = ["workclass", "marital_status", "occupation", "relationship", 
                "race", "sex","native_country", "income"]

for c in cat_features:
    income[c] = pd.Categorical(income[c]).codes

# Train 데이터 세트와 Test 데이터 세트를 구성한다
x = np.array(income)[:, :-1]
y = np.array(income)[:, -1]
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# categorical feature를 multinomial naive bayes로 학습한다.
# --------------------------------------------------------
catN = [1, 3, 4, 5, 6, 7, 9]
catTrainX = trainX[:, catN]
catTestX = testX[:, catN]

# Multinomial model로 categorical Train 데이터 세트를 학습한다.
modelM = MultinomialNB(alpha=1.0)  # alpha = 1.0 : Laplace smoothing (default)
modelM.fit(catTrainX, trainY)

# gaussian feature를 gaussian naive bayes로 학습한다.
# --------------------------------------------------
gauN = [0, 2, 8]
gauTrainX = trainX[:, gauN]
gauTestX = testX[:, gauN]

# Gaussian model로 gaussian Train 데이터 세트를 학습한다.
modelG = GaussianNB()
modelG.fit(gauTrainX, trainY)

# 시험데이터를 이용하여 정확도를 측정한다. 시험데이터도 categorical과 gaussian으로
# 분리돼 있다. 각각의 모형에 따라 확률을 추정한다.
catProb = modelM.predict_proba(catTestX)
gauProb = modelG.predict_proba(gauTestX)

# 두 확률을 곱한다.
mixProb = np.multiply(catProb,gauProb)

# 두 확률의 곱으로 정확도를 측정한다.
mixLabel = np.argmax(mixProb, axis = 1)
accuracy = (testY == mixLabel).mean()
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)
