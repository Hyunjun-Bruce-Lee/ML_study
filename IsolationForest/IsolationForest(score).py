# sklearn의 IsolationForest를 사용해서 anomaly score를 확인한다.
# ------------------------------------------------------------
from sklearn.ensemble import IsolationForest
import numpy as np

# 데이터. shape = (-1, 1) ~ 6행 1열로 맞춘다.
X = np.array([2, 2.5, 3.8, 4.1, 10.5, 15.4]).reshape(-1, 1)

# iForest 모델을 생성한다. score 확인을 위해 tree 개수는 1개로 한다.
# tree가 1개이기 때문에 실행할 때마다 결과가 불안정할 수 있다.
model = IsolationForest(n_estimators=1)

# iForest 모델을 이용하여 데이터 (X)를 학습한다.
model.fit(X)

# 판정 결과를 확인한다. 1 : 정상, -1 : 이상 데이터
pred = model.predict(X)
print("\n판정 결과 : tree = 1개")
print(pred)

# anomaly score를 확인한다.
score = abs(model.score_samples(X))
print("\nAnomaly score :")
print(np.round(score, 3))

# 50개 tree를 사용해서 다시 계산해 본다. 결과가 안정적이다.
model = IsolationForest(n_estimators=50)

# iForest 모델을 이용하여 데이터 (X)를 학습한다.
model.fit(X)

# 판정 결과를 확인한다. 1 : 정상, -1 : 이상 데이터
pred = model.predict(X)
print("\n판정 결과 : tree = 50개")
print(pred)

# anomaly score를 확인한다.
score = abs(model.score_samples(X))
print("\nAnomaly score :")
print(np.round(score, 3))
