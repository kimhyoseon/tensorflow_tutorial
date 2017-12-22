import os
import itertools
import pandas as pd
import tensorflow as tf
from six.moves.urllib.request import urlopen

# tensorflow 윈도우 실행 시 오류 메세지 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 로그에 남을 메시지의 임계 값을 설정
tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/boston')

TRAINING = os.path.join(DATA_PATH, "boston_train.csv")
TRAINING_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/input_fn/boston_train.csv"

TEST = os.path.join(DATA_PATH, "boston_test.csv")
TEST_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/input_fn/boston_test.csv"

PREDICT = os.path.join(DATA_PATH, "boston_predict.csv")
PREDICT_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/r1.4/tensorflow/examples/tutorials/input_fn/boston_predict.csv"

# input_function 정의
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def main(unused_argv):
  # 데이터 다운로드
  if not os.path.exists(DATA_PATH):
    tf.gfile.MakeDirs(DATA_PATH)

  if not os.path.exists(TRAINING):
    raw = urlopen(TRAINING_URL).read()
    with open(TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(TEST):
    raw = urlopen(TEST_URL).read()
    with open(TEST, "wb") as f:
      f.write(raw)

  if not os.path.exists(PREDICT):
    raw = urlopen(PREDICT_URL).read()
    with open(PREDICT, "wb") as f:
      f.write(raw)


  # 데이터 가져오기
  training_set = pd.read_csv(TRAINING, skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv(TEST, skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
  prediction_set = pd.read_csv(PREDICT, skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # x (feature)의 형태 설정
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # 2 layer, fully connected DNN with 10, 10 units 모델 정의
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[10, 10],
                                        model_dir=os.path.join(DATA_PATH, "boston_model"))

  # 훈련
  regressor.train(input_fn=get_input_fn(training_set), steps=5000)

  # 평가
  ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # 예측
  y = regressor.predict(
      input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # .predict() returns an iterator of dicts; convert to a list and print
  # predictions
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()