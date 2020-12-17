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
import nni
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


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
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn



BERT_MODEL = 'uncased_L-12_H-768_A-12'
OUTPUT_DIR = '/research/d3/zmwu/model/nlp_bert/outputs'
VOCAB_FILE = '/research/d3/zmwu/model/nlp_bert/vocab.txt'
CONFIG_FILE = '/research/d3/zmwu/model/nlp_bert/bert_config.json'
INIT_CHECKPOINT = '/research/d3/zmwu/model/nlp_bert/bert_model.ckpt'
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')


params = {
  'respect':"respect",
  'granularity':"core" ,
  'type':"compact",
  'permute':0,
  'offset':0,
  'KMP_BLOCKTIME':"200",
  'OMP_NUM_THREADS':"2",
  'MKL_DYNAMIC':"TRUE",

  'inter_op_parallelism_threads':1,
  'intra_op_parallelism_threads':2,
  'do_common_subexpression_elimination':0,
  'max_folded_constant_in_bytes':10000,
  'do_function_inlining':0,
  'global_jit_level':0,
  'enable_bfloat16_sendrecv':0,
  'infer_shapes':0,
  'place_pruned_graph':0
} 


tuned_params = nni.get_next_parameter() 
params.update(tuned_params) 
t_id = nni.get_trial_id()


#KMP hardware parameter
os.environ["KMP_AFFINITY"] = "verbose,{respect},granularity={specifier},{type},{permute},{offset}".format(
                                respect=params['respect'],
                                specifier=params['granularity'],
                                type=params['type'],
                                permute=params['permute'],
                                offset=params['offset'])
os.environ["KMP_BLOCKTIME"] = params['KMP_BLOCKTIME']
os.environ["OMP_NUM_THREADS"] = params['OMP_NUM_THREADS']
os.environ["MKL_DYNAMIC"] = params['MKL_DYNAMIC']
os.environ["KMP_SETTINGS"] = "TRUE"


# model fix parameter
TRAIN_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 0.001
WARMUP_PROPORTION = 0.05
EVAL_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 128


#data loading
train_df =  pd.read_csv('/research/d3/zmwu/model/nlp_bert/train.csv')
train_df = train_df.sample(1500)
train, test = train_test_split(train_df, test_size = 0.1, random_state=42)
train_lines, train_labels = train.question_text.values, train.target.values
test_lines, test_labels = test.question_text.values, test.target.values
label_list = ['0', '1']
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = create_examples(train_lines, 'train', labels=train_labels)

if params['global_jit_level'] == 0:
    global_jit_level = tf.compat.v1.OptimizerOptions.OFF
elif params['global_jit_level'] == 1:
    global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
else:
    global_jit_level = tf.compat.v1.OptimizerOptions.ON_2

my_config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=int(params['inter_op_parallelism_threads']),
    intra_op_parallelism_threads=int(params['intra_op_parallelism_threads']),
    allow_soft_placement=True,
    log_device_placement=False,
    graph_options=tf.compat.v1.GraphOptions(
        optimizer_options=tf.compat.v1.OptimizerOptions(
            do_common_subexpression_elimination=bool(params['do_common_subexpression_elimination']),
            do_constant_folding=True,
            max_folded_constant_in_bytes=int(params['max_folded_constant_in_bytes']),
            do_function_inlining=bool(params['do_function_inlining']),
            opt_level=tf.compat.v1.OptimizerOptions.L0,
            global_jit_level=global_jit_level
        ),
        enable_bfloat16_sendrecv=bool(params['enable_bfloat16_sendrecv']),
        infer_shapes=bool(params['infer_shapes']),
        place_pruned_graph=bool(params['place_pruned_graph'])
    )
)


run_config = tf.compat.v1.estimator.tpu.RunConfig(
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=None,
    save_checkpoints_secs=None,
    session_config=my_config)


num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  
    use_one_hot_embeddings=True)


estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)


# prepare for train
train_features = run_classifier.convert_examples_to_features(train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
logging.info('***** Started training at {} *****'.format(datetime.datetime.now()))
logging.info('  Num examples = {}'.format(len(train_examples)))
logging.info('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
logging.info("  Num steps = %d", num_train_steps)


##start train
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
logging.info('***** Finished training at {} *****'.format(datetime.datetime.now()))


#prepare for eval
predict_examples = create_examples(test_lines, 'test')
predict_features = run_classifier.convert_examples_to_features(predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)


##eval
result = estimator.predict(input_fn=predict_input_fn)
preds = []
for prediction in tqdm(result):
    for class_probability in prediction['probabilities']:
      preds.append(float(class_probability))
results = []
for i in tqdm(range(0,len(preds),2)):
  if preds[i] < 0.9:
    results.append(1)
  else:
    results.append(0)


# calculate the result:
logging.info(accuracy_score(np.array(results), test_labels))
logging.info(f1_score(np.array(results), test_labels))
final_acc = accuracy_score(np.array(results), test_labels)
nni.report_final_result(float(final_acc))