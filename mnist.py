import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np

# tensorflow 윈도우 실행 시 오류 메세지 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST 예제 (Modified National Institute of Standards and Technology)

# 학습 및 평가 데이터 다운로드
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)
# Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object
print(mnist.train.images)
# [[ 0.  0.  0. ...,  0.  0.  0.] <= 28 * 28 = 784
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# ...,
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]]
# train = 50,000개, test = 10,000개, validation = 5,000개
print(mnist.train.labels)
# [[ 0.  1.  0. ...,  0.  0.  0.] <= 10
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  1.  0.]
# ...,
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  0. ...,  0.  0.  0.]
# [ 0.  0.  1. ...,  0.  0.  0.]]
# train = 50,000개, test = 10,000개, validation = 5,000개


# 구성 단계(construction phase)

# 학습 데이터를 넣을 변수 (빈값), None은 제한없음 (무한대)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 계산 결과 y값
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 소프트맥스란, 여러 개의 보기의 정답일 확률의 합이 1
# 예를 들어 5지선다 문제의 정답 확률을 아래와 같이 표현
# 1번이 정답일 확률 - 0
# 2번이 정답일 확률 - 0.1
# 3번이 정답일 확률 - 0.2
# 4번이 정답일 확률 - 0.7
# 5번이 정답일 확률 - 0

# 학습으로 생성될 W, b 변수 (0으로 세팅)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# 실행 단계(execution phase)

# cross_entropy (교차 엔트로피) : 소프트맥스 처리 된 계산값 y와 실제 정답인 y_에 대한 loss 계산시 사용
# reduce_mean : 벡터 안 모든 수의 평균값
# reduce_sum : 벡터 안 모든 수의 합

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
# cost = tf.reduce_mean(cross_entropy)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 세션 생성
sess = tf.Session()

# 변수 초기화
sess.run(tf.global_variables_initializer())

# 학습
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

# 결과 확인
# 정답배열중에서 가장 큰 값의 index가 같은지 비교
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# [1,0,1,1] 정답은1, 오답은 0

# 정답률
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))