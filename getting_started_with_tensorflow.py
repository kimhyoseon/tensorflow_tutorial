import os
import tensorflow as tf

# tensorflow 윈도우 실행 시 오류 메세지 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 구성 단계(construction phase)

# 변수1
W = tf.Variable([.3], dtype=tf.float32)
# 변수2
b = tf.Variable([-.3], dtype=tf.float32)
# 변수3 (빈값)
x = tf.placeholder(tf.float32)
# 선형 모델 연산 노드
linear_model = W * x + b

# 실행 단계(execution phase)

# 세션
sess = tf.Session()

# 모든 변수를 초기화 (tf.Variable에 설정된 값으로 세팅)
init = tf.global_variables_initializer()
sess.run(init)

# 선형 모델을 테스트 실행
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
# print [ 0.          0.30000001  0.60000002  0.90000004]
# 0.3 * 1 - 0.3 = 0
# 0.3 * 2 - 0.3 = 0.3
# 0.3 * 3 - 0.3 = 0.6
# 0.3 * 4 - 0.3 = 0.9

# 위에서 나온 y값이 아닌 원하는 y값 생성
y = tf.placeholder(tf.float32)

# 계산된 y값과 지정된 y값의 차이를 제곱
squared_deltas = tf.square(linear_model - y)

# 결과 배열을 모두 sum
loss = tf.reduce_sum(squared_deltas)

# loss 출력
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# print 23.66
# (0 - 0)^2 + (0.3 + 1)^2 + (0.6 + 2)^2 + (0.9 + 3)^2 = 23.66
# 0 + 1.69 + 6.76 + 15.21 = 23.66
# W = 0.3, b = - 0.3 으로 설정된 선형 모델에서는 loss가 23.66
# W = -1, b = 1 인 경우 loss가 0이 나오고 정답
# W = -1, b = 1의 정답을 학습을 통해 찾아내는 것이 머신러닝

# 최적의 값을 찾는 optimizer (GradientDescentOptimizer (기울기 하강) 사용)
# 위의 loss이 낮아지도록 자동으로 학습
optimizer = tf.train.GradientDescentOptimizer(0.01) # 0.01은 학습속도 (learning_rate)
train = optimizer.minimize(loss)

# 변수를 다시 초기화
sess.run(init)

# 1000회 학습
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# 변수 W와 b print
print(sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
# W = -0.9999969, b = 0.99999082, loss = 5.6999738e-11
# W = -1, b = 1, loss = 0 에 매우 가까우므로 정답을 학습을 통해 알아내는데 성공