seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import numpy as np
np.random.seed(seed_value)
import random
random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

import pandas as pd
import sys
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import modeling
import optimization
import run_classifier
import tokenization
from tqdm import tqdm
import time
import shutil
from absl import flags, app
import logging
from selftf.lib.mltuner.mltuner_util import MLTunerUtil


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )


mltunerUtil = MLTunerUtil()

def create_examples(lines, set_type, labels=None):
#Generate data for the BERT model
    guid = f'{set_type}'
    examples = []
    if guid == 'train':
        for line, label in zip(lines, labels):
            text_a = line
            label = str(label)
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    else:
        for line in lines:
            text_a = line
            label = '0'
            examples.append(
              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = 16
    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.shard(mltunerUtil.get_num_worker(),mltunerUtil.get_worker_index())
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
      d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    else:
      d = d.shard(mltunerUtil.get_num_worker(),mltunerUtil.get_worker_index())
      d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def main(argv):

  BERT_MODEL = 'uncased_L-12_H-768_A-12'
  VOCAB_FILE = '/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/bert_tf2/vocab.txt'
  CONFIG_FILE = '/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/bert_tf2/bert_config.json'
  INIT_CHECKPOINT = '/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/bert_tf2/bert_model.ckpt'
  DO_LOWER_CASE = BERT_MODEL.startswith('uncased')
  model_dir = "{}/{}".format("/opt/tftuner", mltunerUtil.get_job_id())

  # model fix parameter
  TRAIN_BATCH_SIZE = mltunerUtil.get_batch_size()
  NUM_TRAIN_EPOCHS = 3
  LEARNING_RATE = mltunerUtil.get_learning_rate()
  WARMUP_PROPORTION = 0.05
  EVAL_BATCH_SIZE = 8
  MAX_SEQ_LENGTH = 128


  #data loading
  train_df =  pd.read_csv('/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/bert_tf2/train.csv')
  train_df = train_df.sample(1000)
  train, test = train_test_split(train_df, test_size = 0.1, random_state=42)
  train_lines, train_labels = train.question_text.values, train.target.values
  test_lines, test_labels = test.question_text.values, test.target.values
  label_list = ['0', '1']
  tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
  train_examples = create_examples(train_lines, 'train', labels=train_labels)


  num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
  num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


  strategy = tf.distribute.experimental.ParameterServerStrategy()
  session_config = mltunerUtil.get_tf_session_config()
  config = tf.compat.v1.estimator.tpu.RunConfig(
    train_distribute=strategy,
    model_dir=model_dir,
    save_checkpoints_steps=None,
    save_checkpoints_secs=None,
    session_config=session_config)

  model_fn = run_classifier.model_fn_builder(
      bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
      num_labels=len(label_list),
      init_checkpoint=None,
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  
      use_one_hot_embeddings=True)


  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
      model_fn=model_fn,
      config=config,
      train_batch_size=TRAIN_BATCH_SIZE,
      eval_batch_size=EVAL_BATCH_SIZE)

  class LoggerHook(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self):
        self.last_run_timestamp = time.time()
    
    def after_run(self, run_context, run_values):
        session: tf.Session = run_context.session
        loss, step = session.run([tf.compat.v1.get_collection("losses")[0],
                                  tf.compat.v1.get_collection("global_step_read_op_cache")[0]])
        logging.debug("step:{} loss:{}".format(step, loss))
        mltunerUtil.report_iter_loss(step, loss,
                                     time.time() - self.last_run_timestamp)
        self.last_run_timestamp = time.time()

  # prepare for train
  train_features = run_classifier.convert_examples_to_features(train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  train_input_fn = input_fn_builder(
      features=train_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=True,
      drop_remainder=True)

  predict_examples = create_examples(test_lines, 'test')
  predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = input_fn_builder(
      features=predict_features,
      seq_length=MAX_SEQ_LENGTH,
      is_training=False,
      drop_remainder=False)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,hooks=[LoggerHook()])
  eval_spec = tf.estimator.EvalSpec(input_fn=predict_input_fn)

  # wait for chief ready?
  if not (mltunerUtil.is_chief() or mltunerUtil.is_ps()):
      time.sleep(1)
      if not tf.io.gfile.exists(model_dir):
          logging.debug("wait for chief init")
          time.sleep(1)
  tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)

#   # start training
#   logging.info('***** Started training at {} *****'.format(datetime.datetime.now()))
#   logging.info('  Num examples = {}'.format(len(train_examples)))
#   logging.info('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
#   logging.info("  Num steps = %d", num_train_steps)
#   estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
#   logging.info('***** Finished training at {} *****'.format(datetime.datetime.now()))

#   # start eval
#   result = estimator.predict(input_fn=predict_input_fn)
#   preds = []
#   for prediction in tqdm(result):
#       for class_probability in prediction['probabilities']:
#         preds.append(float(class_probability))
#   results = []
#   for i in tqdm(range(0,len(preds),2)):
#     if preds[i] < 0.9:
#       results.append(1)
#     else:
#       results.append(0)

#   # calculate the result:
#   logging.info("accuracy:{}".format(accuracy_score(np.array(results), test_labels)))
#   logging.info("f1_score:{}".format(f1_score(np.array(results), test_labels)))



if __name__ == '__main__':
  app.run(main)
