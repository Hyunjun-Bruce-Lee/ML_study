# dataset : https://www.kaggle.com/mlg-ulb/creditcardfraud
# --------------------------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Credit data set을 읽어온다
# 사용 시간 (Time), 사용 금액 (Amount)을 정상 여부 (Class)를 
# 제외한 나머지 feature들은 PCA로 변환된 수치임.
# Class (0 : 정상 사용, 1 : 비정상 사용 (fraud))
df = pd.read_csv('dataset/creditcard(fraud).csv')

# 사용 시간대별 트랜잭션의 분포를 확인해 본다.
f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(10, 6))
ax1.hist(df['Time'][df['Class'] == 0], bins=50)
ax2.hist(df['Time'][df['Class'] == 1], bins=50)
plt.xlabel('Time')
plt.ylabel('Transactions')
ax1.set_title('Normal use')
ax2.set_title('Abnormal use')
plt.show()

# 사용 금액별 트랜잭션의 분포를 확인해 본다
f, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(10,6))
ax1.hist(df['Amount'][df['Class'] == 0], bins=50)
ax2.hist(df['Amount'][df['Class'] == 1], bins=50)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.set_title('Normal use')
ax2.set_title('Abnormal use')
plt.xlabel('Amount ($)')
plt.ylabel('Transactions')
plt.show()

# feature별 분포를 확인해 본다.
feature = 'V11'
sns.distplot(df[feature][df['Class'] == 0], bins=50)
sns.distplot(df[feature][df['Class'] == 1], bins=50)
plt.legend(['Normal','Abnormal'], loc='best')
plt.title(feature + ' distribution')
plt.show()
       
# 학습 데이터를 만든다.
credit = np.array(df)
trainX = credit[:, :-1]
trainY = credit[:, -1]

# 비정상 사례의 비율을 확인해 본다.
normal = (trainY == 0).sum()
fraud = (trainY == 1).sum()
print("\n정상 사례 = ", normal)
print("비정상 사례 = ", fraud)
print("비정상 사례 비율 = %.4f (%%)" % (100 * fraud / normal))

# iForest로 이상치를 확인한다.
model = IsolationForest(n_estimators = 100)

# iForest 모델을 이용하여 데이터를 학습한다.
model.fit(trainX)

# Anomaly score를 확인한다.
score = abs(model.score_samples(trainX))

plt.hist(score, bins = 50)
plt.title('distribution of anomaly score')
plt.xlabel('anomaly score')
plt.ylabel('frequency')
plt.show()

# Anomaly score가 1에 가까운 데이터를 이상 데이터로 판정한다.
predY = (score > 0.65).astype(int)
fraud_count = (predY == 1).sum()
print('이상 데이터로 판정한 개수 =', fraud_count)

# confusion matrix를 확인한다.
cm = confusion_matrix(trainY, predY)
cm_df = pd.DataFrame(cm)
cm_df.columns = ['pred_normal', 'pred_abnormal']
cm_df.index = ['actual_normal', 'actual_abnormal']
print('confusion matrix :')
print(cm_df)

# accuracy는 큰 의미가 없고, precision과 recall이 의미있음.
print('\n비정상으로 판정한 것 중에,')
print('정상 데이터를 비정상으로 판정한 비율 = %.4f' % (cm[0, 1] / cm[:, 1].sum()))
print('비정상 데이터를 비정상으로 판정한 비율 = %.4f' % (cm[1, 1] / cm[:, 1].sum()))

# 추가 측정치
print('accuracy = %.4f' % accuracy_score(trainY, predY))
print(confusion_matrix(trainY, predY))
print(classification_report(trainY, predY))

# contamination으로 비정상 데이터 판정 비율을 설정하려면 아래처럼 한다.
#model = IsolationForest(n_estimators = 100, contamination=0.01)
#predY = model.predict(trainY)  # 정상 = 1, 비정상 = -1