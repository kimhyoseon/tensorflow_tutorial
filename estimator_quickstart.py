import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# tensorflow 윈도우 실행 시 오류 메세지 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Data sets
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/iris')

IRIS_TRAINING = os.path.join(DATA_PATH, "iris_training.csv")
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = os.path.join(DATA_PATH, "iris_test.csv")
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # 데이터 파일이 저장되어 있지 않다면 데이터파일 다운로드 후 파일 저장
  # 데이터 샘플은 6.4, 2.8, 5.6, 2.2, 2 형태이며 앞 4개는 x, 뒤 1개는 y
  if not os.path.exists(DATA_PATH):
    tf.gfile.MakeDirs(DATA_PATH)

  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

  # 데이터 가져오기
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # x의 형태 설정
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # estimator를 이용해 DNNClassifier 설정
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir=os.path.join(DATA_PATH, "iris_model"))
  # 훈련 데이터 정의
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # 훈련 2000회 실시
  classifier.train(input_fn=train_input_fn, steps=2000)

  # 테스트 데이터 정의
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # 평가
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # 2개의 샘플로 정답 예측
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.6, 2.8, 4.9, 2.0]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()