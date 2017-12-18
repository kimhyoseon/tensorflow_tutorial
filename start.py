import os
import tensorflow as tf

# tensorflow 윈도우 실행 시 오류 메세지 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 텐서플로어 2단계
# 구성 단계(construction phase) : 노드와 그래프를 만들어 조립
# 실행 단계(execution phase) : 세션을 이용해 그래프를 실행

# 텐서란?
# 이미지, 소리, 텍스트 등은 컴퓨터가 이해할 수 없는 형태
# 이미지, 소리, 텍스트 등을 컴퓨터가 이해할 수 있게 의미있는 숫자의 다차원 배열로 변환한 값
# ex)
# 3 # a rank 0 tensor; a scalar with shape []
# [1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

# 노드란?
# (전문 용어 특히 컴퓨터) (연결망의) 교점[접속점]
# a network node
# 통신망의 교점
# 변수, 상수에 설정한다거나 덧셈, 뺄샘등의 연산을 한다거나 하는 하나의 단위

# float 상수를 설정하는 노드 / 3.0
node_constant = tf.constant(3.0, dtype=tf.float32)

# float 변수를 설정하는 노드 / 추후에 값을 넣어 줌
node_variable = tf.placeholder(tf.float32)

# 덧셈을 연산하는 노드
node_add = node_constant + node_variable

# 그래프란?
# 노드를 선으로 연결한 모양
# 노드 하나하나가 실행되는 과정을 선으로 연결한 과정 전체
# 위의 node_constant, node_variable, node_add 전 과정

# 세션
sess = tf.Session()

# 세션에서 그래프 실행
print(sess.run(node_add, {node_variable: 10.0}))
# result : 13.0