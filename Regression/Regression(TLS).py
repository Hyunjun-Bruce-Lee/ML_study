# TensorFlow 2.0을 이용하여 직선회귀를 연습한다.
# 직선회귀 방법 : Total Least Square (직교 회귀 : Orthogonal Distance)
# ------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

# 샘플 데이터 1,000개를 생성한다
# y = ax + b + e
def createData(a, b, n):
   resultX = []
   resultY = []
   for i in range(n):
       x = np.random.normal(0.0, 0.5)
       y = a * x + b + np.random.normal(0.0, 0.05)
       resultX.append(x)
       resultY.append(y)
       
   return resultX, resultY

# inputY = 0.1 * inputX + 0.3 + 잔차
x, y = createData(0.1, 0.3, 1000)

# 선형 추정 식을 정의한다
# predY = W * inputX + b
r = tf.random.uniform([1], -1.0, 1.0)
W = tf.Variable(r, name = "W")
b = tf.Variable(tf.zeros([1]), name = "Bias")

# 학습할 optimizer를 정의한다
opt = optimizers.Adam(learning_rate = 0.05)

# 점 (x, y)와 회귀직선 사이의 수직 거리
def loss(x):
    bunja = tf.abs(tf.subtract(tf.add(tf.multiply(W, x), b), y))  # 분자
    bunmo = tf.sqrt(tf.add(tf.square(W), tf.constant(1.0)))       # 분모
    
    # Loss function을 정의한다. (MSE : Mean Square Error)
    return tf.reduce_mean(tf.square(tf.divide(bunja, bunmo)))

    
# 학습한다
trLoss = []
for i in range(300):
    opt.minimize(lambda: loss(x), var_list = [W, b])
    trLoss.append(loss(x))
    
    if i % 10 == 0:
        print("%d) %f" % (i, trLoss[-1]))

# 결과를 확인한다
print("\n*회귀직선의 방정식 (TLS) : y = %.4f * x +  %.4f" % (W.numpy(), b.numpy()))
yHat =  W.numpy() * x + b.numpy()

fig = plt.figure(figsize=(10, 4))
p1 = fig.add_subplot(1,2,1)
p2 = fig.add_subplot(1,2,2)

p1.plot(x, y, 'ro', markersize=1.5)
p1.plot(x, yHat)

p2.plot(trLoss, color='red', linewidth=1)
p2.set_title("Loss function")
p2.set_xlabel("epoch")
p2.set_ylabel("loss")
plt.show()

    