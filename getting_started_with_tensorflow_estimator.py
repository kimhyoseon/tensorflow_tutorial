import os
import tensorflow as tf
import numpy as np

# tensorflow 윈도우 실행 시 오류 메세지 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# estimator에서 제공하는 모델로 더 간단하게 학습이 가능
# 학습할 데이터의 feature를 지정
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 학습 데이터
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
# 평가 데이터
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# estimator에 넣을 tf 데이터 타입으로 변환
input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=500, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=500, shuffle=False)

# 선형회귀 모델을 사용
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# 학습 시작
estimator.train(input_fn=input_fn, steps=500)

# 평가
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print(train_metrics)
print(eval_metrics)
# {'average_loss': 0.00059131969, 'loss': 0.0023652788, 'global_step': 500}
# {'average_loss': 0.0088197431, 'loss': 0.035278972, 'global_step': 500}