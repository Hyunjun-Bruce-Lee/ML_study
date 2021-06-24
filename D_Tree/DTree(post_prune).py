# Decision Tree로 income 데이터를 학습한다 (post-prnning).
# 데이터 세트 : http://archive.ics.uci.edu/ml/datasets/Adult
# ------------------------------------------------------------------------------------
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

path = DecisionTreeClassifier().cost_complexity_pruning_path(trainX, trainY)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

plt.figure(figsize=(8,4))
plt.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
plt.xlabel("effective alpha")
plt.ylabel("total impurity of leaves")
plt.title("Total Impurity vs effective alpha for training set")

# ccp_alphas가 너무 작은 것은 제외한다.
ccp_alphas = ccp_alphas[np.where(ccp_alphas > 0.0001)]

clfs = []
for i, ccp_alpha in enumerate(ccp_alphas):
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    clf.fit(trainX, trainY)
    clfs.append(clf)
        
    print('%d) ccp_alphas = %.4f done.' % (i, ccp_alpha))

print('마지막 tree의 노드 개수 = %d' % clfs[-1].tree_.node_count)
print('마지막 tree의 alpha = %.4f' % ccp_alphas[-1])
print('마지막 tree의 depth = %d' % clfs[-1].tree_.max_depth)

# 마지막 tree는 depth = 0이므로 제외한다.
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# ccp_alphas가 증가할수록 node개수와 depth가 감소하는 것을 확인한다.
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]

plt.figure(figsize=(8,4))
plt.plot(ccp_alphas, node_counts, marker='o')
plt.xlabel('ccp_alphas')
plt.ylabel('node_counts')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(ccp_alphas, depth, marker='o')
plt.xlabel('ccp_alphas')
plt.ylabel('depth')
plt.show()

# ccp_alphas를 적용한 tree들 (clfs)로 score를 계산한다.
# clfs는 앞 부분 n개만 사용한다. 뒷 부분은 alpha가 너무 크기 때문에 제외한다.
n=70
train_scores = [clf.score(trainX, trainY) for clf in clfs[:n]]
test_scores = [clf.score(testX, testY) for clf in clfs[:n]]

plt.figure(figsize=(8,5))
plt.plot(ccp_alphas[:n], train_scores[:n], marker='o')
plt.plot(ccp_alphas[:n], test_scores[:n], marker='o')
plt.xlabel('ccp_alphas')
plt.ylabel('score')
plt.show()

# test_scores[:n]중 가장 큰 최적 alpha를 찾는다.
opt_alpha = ccp_alphas[np.argmax(test_scores[:n])]

# opt_alpha를 적용한 tree를 사용한다.
dt = DecisionTreeClassifier(ccp_alpha=opt_alpha)
dt.fit(trainX, trainY)

print('시험 데이터의 정확도 = %.4f' % dt.score(testX, testY))
print('Optimal alpha = %.8f' % opt_alpha)

# feature들의 중요도를 분석한다.
feature_importance = dt.feature_importances_
feature_name = list(income.columns)
n_feature = trainX.shape[1]
idx = np.arange(n_feature)

plt.barh(idx, feature_importance, align='center')
plt.yticks(idx, feature_name, size=12)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
