from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from google.cloud import translate_v2 as translate
import six
from collections import OrderedDict


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files) for the task.")

flags.DEFINE_string("bert_config_file", None, "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")

flags.DEFINE_string("bert_config_file_en", None, "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")

flags.DEFINE_string("bert_config_file_ch", None, "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("vocab_file_en", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("vocab_file_ch", None, "The vocabulary file that the BERT model was trained on.")


flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("init_checkpoint_en", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("init_checkpoint_ch", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

tf.flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_string("translation_key", None, "google translation API json key file.")

tf.flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")

tf.app.flags.DEFINE_string('f', '', 'kernel')


def get_examples(phase):
  file = phase+".tsv"
  file_name = os.path.join(FLAGS.data_dir, file)
  with open(file_name, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    examples = []
    for i, line in enumerate(reader):
      if i == 0:
        continue
      feature = dict()
      try:
        text_a = line[1]
        text_b = None
        label = line[2]
      except IndexError:
        continue
      feature['text_a'] = text_a
      feature['text_b'] = text_b
      feature['label'] = label
      examples.append(feature)
    return examples



def create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f



def translate_text(line, lang):
  translate_client = translate.Client()
  if isinstance(line, six.binary_type):
    text = line.decode('utf-8')
  else:
    text = line
  result = translate_client.translate(text, target_language=lang)
  return result['translatedText']


def write_features(examples, output_file, tokenizers):
  features = []
  writer = tf.python_io.TFRecordWriter(output_file)

  label_list = [example['label'] for example in examples]

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  for example in examples:
    if example == None:
      feature = OrderedDict()
      feature['input_ids']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['input_mask']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['segment_ids']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['input_ids_en']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['input_mask_en']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['segment_ids_en']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['input_ids_ch']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['input_mask_ch']=create_int_feature([0] * FLAGS.max_seq_length)
      feature['segment_ids_ch']=create_int_feature([0] * FLAGS.max_seq_length)

      feature['label_id']=create_int_feature([0])
      feature['is_real_example']=create_int_feature([int(False)])
    else:
      # do translation
      eng_text_a = translate_text(example['text_a'], 'en')
      chi_text_a = translate_text(example['text_a'], 'zh-TW')
      eng_text_b = None
      chi_text_b = None
      if example['text_b']:
        eng_text_b = translate_text(example['text_b'], 'en')
        chi_text_b = translate_text(example['text_b'], 'zh-TW')

      all_lines = []
      results = []
      all_lines.append([example['text_a'], example['text_b']])
      all_lines.append([eng_text_a, eng_text_b])
      all_lines.append([chi_text_a, chi_text_b])
      for i, [tokens_a, tokens_b] in enumerate(all_lines):
        tokens_a = tokenizers[i].tokenize(example['text_a'])
        tokens_b = None
        if example['text_b']:
          tokens_b = tokenizers[i].tokenize(example['text_b'])

        if tokens_b:
          _truncate_seq_pair(tokens_a, tokens_b, FLAGS.max_seq_length - 3)
        else:
          if len(tokens_a) > FLAGS.max_seq_length - 2:
            tokens_a = tokens_a[0:(FLAGS.max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
          for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
          tokens.append("[SEP]")
          segment_ids.append(1)

        input_ids = tokenizers[i].convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < FLAGS.max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)

        assert len(input_ids) == FLAGS.max_seq_length
        assert len(input_mask) == FLAGS.max_seq_length
        assert len(segment_ids) == FLAGS.max_seq_length
        results.append([input_ids, input_mask, segment_ids])

      label_id = label_map[example['label']]
      feature = OrderedDict()
      feature['input_ids']=create_int_feature(results[0][0])
      feature['input_mask']=create_int_feature(results[0][1])
      feature['segment_ids']=create_int_feature(results[0][2])

      feature['input_ids_en']=create_int_feature(results[1][0])
      feature['input_mask_en']=create_int_feature(results[1][1])
      feature['segment_ids_en']=create_int_feature(results[1][2])

      feature['input_ids_ch']=create_int_feature(results[2][0])
      feature['input_mask_ch']=create_int_feature(results[2][1])
      feature['segment_ids_ch']=create_int_feature(results[2][2])

      feature['label_id']=create_int_feature([label_id])
      feature['is_real_example']=create_int_feature([int(True)])
      tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(tf_example.SerializeToString())
  writer.close()

  return label_list




def _truncate_seq_pair(tokens_a, tokens_b, max_length):
while True:
  total_length = len(tokens_a) + len(tokens_b)
  if total_length <= max_length:
    break
  if len(tokens_a) > len(tokens_b):
    tokens_a.pop()
  else:
    tokens_b.pop()

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
         labels, num_labels, use_one_hot_embeddings):
"""Creates a classification model."""
model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=use_one_hot_embeddings)


output_layer = model.get_pooled_output()

hidden_size = output_layer.shape[-1].value

output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

with tf.variable_scope("loss"):
  if is_training:
    # I.e., 0.1 dropout
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

  logits = tf.matmul(output_layer, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  probabilities = tf.nn.softmax(logits, axis=-1)
  log_probs = tf.nn.log_softmax(logits, axis=-1)

  one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

  per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
  loss = tf.reduce_mean(per_example_loss)




  return (loss, per_example_loss, logits, probabilities)


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_ids_en": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask_en": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids_en": tf.FixedLenFeature([seq_length], tf.int64),
    "input_ids_ch": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask_ch": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids_ch": tf.FixedLenFeature([seq_length], tf.int64),
    "label_id": tf.FixedLenFeature([], tf.int64),
    "is_real_example": tf.FixedLenFeature([], tf.int64),
}

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
    t = tf.to_int32(t)
    example[name] = t

  return example

def input_fn(params):
  """The actual input function."""
  batch_size = params["batch_size"]

  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  d = tf.data.TFRecordDataset(input_file)
  if is_training:
    d = d.repeat()
    d = d.shuffle(buffer_size=100)

  d = d.apply(
    tf.contrib.data.map_and_batch(
      lambda record: _decode_record(record, name_to_features),
      batch_size=batch_size,
      drop_remainder=drop_remainder))

  return d

return input_fn


def model_fn_builder(bert_configs, num_labels, checkpoints, learning_rate, num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

  tf.logging.info("*** Features ***")
  for name in sorted(features.keys()):
    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

  input_ids = features["input_ids"]
  input_mask = features["input_mask"]
  segment_ids = features["segment_ids"]
  label_ids = features["label_id"]
  is_real_example = None
  if "is_real_example" in features:
    is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
  else:
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  with tf.variable_scope("multi"):
    (total_loss, per_example_loss, logits, probabilities) = create_model(bert_configs[0], is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)
  with tf.variable_scope("english"):
    (total_loss_en, per_example_loss_en, logits_en, probabilities_en) = create_model(bert_configs[1], is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)
  with tf.variable_scope("chinese"):
    (total_loss_ch, per_example_loss_ch, logits_ch, probabilities_ch) = create_model(bert_configs[2], is_training, input_ids, input_mask, segment_ids, label_ids, num_labels, use_one_hot_embeddings)

  tvars = tf.trainable_variables()
  initialized_variable_names = {}
  scaffold_fn = None
  scopes = ['multi', 'english', 'chinese']
  for i, init_checkpoint in enumerate(checkpoints):
    if init_checkpoint:
      (assignment_map, initialized_variable_names_part) = modeling.get_assignment_map_from_checkpoint_with_scope(tvars, init_checkpoint, scopes[i])
      if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
      else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    initialized_variable_names = {**initialized_variable_names, **initialized_variable_names_part}

  tf.logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
    init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

  output_spec = None
  if mode == tf.estimator.ModeKeys.TRAIN:

    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn)
  elif mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(per_example_loss, label_ids, logits, is_real_example):
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions, weights=is_real_example)
    loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
    return {"eval_accuracy": accuracy, "eval_loss": loss,}

    eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, is_real_example])
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=total_loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
  else:
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions={"probabilities": probabilities}, scaffold_fn=scaffold_fn)
  return output_spec

return model_fn




def main(_):
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = FLAGS.translation_key

  task = 'hate'
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  bert_config_en = modeling.BertConfig.from_json_file(FLAGS.bert_config_file_en)
  bert_config_ch = modeling.BertConfig.from_json_file(FLAGS.bert_config_file_ch)

  tpu_cluster_resolver = None
  train_file = "train_record.tf_record"
  examples = get_examples('train')
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  tokenizer_multi = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  tokenizer_en = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file_en, do_lower_case=FLAGS.do_lower_case)
  tokenizer_ch = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file_ch, do_lower_case=FLAGS.do_lower_case)
  tokenizers = [tokenizer_multi, tokenizer_en, tokenizer_ch]
  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  if FLAGS.do_train:
    num_train_steps = int(len(examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


  label_list = write_features(examples, train_file, tokenizers)
  checkpoints = [FLAGS.init_checkpoint, FLAGS.init_checkpoint_en, FLAGS.init_checkpoint_ch]
  configs = [bert_config, bert_config_en, bert_config_ch]
  model_fn = model_fn_builder(bert_configs=configs, num_labels=len(label_list), checkpoints=checkpoints, learning_rate=FLAGS.learning_rate, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, use_tpu=FLAGS.use_tpu, use_one_hot_embeddings=FLAGS.use_tpu)

  run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver, master=FLAGS.master, model_dir=FLAGS.output_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps, tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop, num_shards=FLAGS.num_tpu_cores, per_host_input_for_training=is_per_host))

  estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu, model_fn=model_fn, config=run_config, train_batch_size=FLAGS.train_batch_size, eval_batch_size=FLAGS.eval_batch_size, predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_input_fn = file_based_input_fn_builder(input_file=train_file, seq_length=FLAGS.max_seq_length, is_training=True, drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  
  if FLAGS.do_eval:
    examples = get_examples('eval')

    num_actual_eval_examples = len(examples)

    if FLAGS.use_tpu:
      while len(examples) % FLAGS.eval_batch_size != 0:
      examples.append(None)
    
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    label_list = write_features(examples, train_file, tokenizers)

    eval_steps = None

    if FLAGS.use_tpu:
      assert len(examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(examples) // FLAGS.eval_batch_size)
    
    eval_drop_remainder = True if FLAGS.use_tpu else False
    label_list = write_features(examples, eval_file, tokenizers)
    eval_input_fn = file_based_input_fn_builder(input_file=eval_file, seq_length=FLAGS.max_seq_length, is_training=True, drop_remainder=True)
    
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
      tf.logging.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))
    

if __name__ == "__main__":
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("vocab_file")
flags.mark_flag_as_required("vocab_file_en")
flags.mark_flag_as_required("vocab_file_chi")
flags.mark_flag_as_required("bert_config_file")
flags.mark_flag_as_required("bert_config_file_en")
flags.mark_flag_as_required("bert_config_file_ch")
flags.mark_flag_as_required("translation_key")
flags.mark_flag_as_required("output_dir")
tf.app.run()