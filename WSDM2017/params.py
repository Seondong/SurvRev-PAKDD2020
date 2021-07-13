import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('binary', True, 'Task type')
flags.DEFINE_integer('max_trajectory_len', 20, 'Maximum trajectory length in each visit')
flags.DEFINE_integer('max_num_histories', 5, 'Maximum number of histories used for RNN')
flags.DEFINE_integer('emb_dim_uid', 32, 'Dimension of embedded user id')
flags.DEFINE_integer('emb_dim_zone', 32, 'Dimension of embedded zone')
flags.DEFINE_integer('batch_size', 32, 'Batch size of the model')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer')
flags.DEFINE_string('result_directory', 'results', 'Directory to put intermediary results')
flags.DEFINE_string('pre_release_path', '../data_sample/indoor/', 'Pre-release path: sample data')
flags.DEFINE_string('release_path', '../data/indoor/', 'Release path: benchmark data')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('last_timestamp_A', 1514725408, 'The last unix timestamp of Indoor/Store_A raw data')
flags.DEFINE_integer('last_timestamp_B', 1514731520, 'The last unix timestamp of Indoor/Store_B raw data')
flags.DEFINE_integer('last_timestamp_C', 1509715252, 'The last unix timestamp of Indoor/Store_C raw data')
flags.DEFINE_integer('last_timestamp_D', 1510149991, 'The last unix timestamp of Indoor/Store_D raw data')
flags.DEFINE_integer('last_timestamp_E', 1510148006, 'The last unix timestamp of Indoor/Store_E raw data')

