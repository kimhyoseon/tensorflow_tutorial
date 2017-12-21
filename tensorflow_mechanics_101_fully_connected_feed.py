import os
import sys
import time
import argparse
import tensorflow as tf
import tensorflow_mechanics_101 as mnist
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import xrange

FLAGS = None

# 로그를 모두 지움
def delete_log_data():
  if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

# x, y의 placeholder 정의
def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

# 학습할 dict (1회분)
def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

# 평가 데이터 print
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

# 학습 시작
def run_training():
  # 학습 데이터
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  # 그래프 정의
  with tf.Graph().as_default():
    # x, y의 placeholder 정의
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # 추론 함수 실행 후 logits (계산에 의해 추론된 y) 획득
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # 손실 함수 실행 후 손실 획득
    loss = mnist.loss(logits, labels_placeholder)

    # 훈련 함수에 손실값을 넣어 operation 획득
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # 평가 함수
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # ?? Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # 변수 초기화
    init = tf.global_variables_initializer()

    # 훈련 체크포인트를 로그로 저장하기 위한 Saver
    saver = tf.train.Saver()

    # 세션 생성
    sess = tf.Session()

    # 써머리 파일저장을 위한 summary_writer 초기화
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # 학습루프 시작
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # 1회 학습할 x데이터와 y데이터 dictionary
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # 학습 실행
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # 100회마다 써머리 저장
      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # 써머리 파일 갱신
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # 1000회 마다 평가 결과 저장
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)

# 메인 함수
def main(_):
    delete_log_data()
    run_training()


# tensorflow_mechanics_101_fully_connected_feed.py가 직접 실행되는 경우
if __name__ == '__main__':

  # argparse를 사용하여 파라미터 정의
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data/input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  # 정의된 파라미터 -> FLAGS
  FLAGS, unparsed = parser.parse_known_args()
  print(FLAGS)
  # Namespace(
  #   batch_size=100,
  #   fake_data=False,
  #   hidden1=128,
  #   hidden2=32,
  #   input_data_dir='/tmp\\tensorflow/mnist/input_data',
  #   learning_rate=0.01,
  #   log_dir='/tmp\\tensorflow/mnist/logs/fully_connected_feed',
  #   max_steps=2000)
  print(unparsed)
  # []

  tf.app.run(main=main)