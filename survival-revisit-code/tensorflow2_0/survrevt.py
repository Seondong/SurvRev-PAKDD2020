import numpy as np
import tensorflow as tf
from tensorflow import keras
from params import FLAGS
from data import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

"""
* Filename: survrevt.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Tensorflow implementation. (Not implemented yet - Implement if needed)
"""


class SurvRevT():
    """Class SurvRevT describes our Tensorflow model - Not implemented yet"""
    def __init__(self, gpu_id, input_file, log_dir):
        self.gpu_id = gpu_id
        self.input_file = input_file
        self.store_id = input_file[-1]
        self.log_dir = log_dir

        # input data
        self.train_data = None
        self.test_data = None

        # model initialization
        self.num_handcrafted_feature = 15  # Todo: Will be calculated later
        self.x_handcraft = tf.placeholder("float", shape=[None, self.num_handcrafted_feature], name="x_handcraft")

        d_num_zones = {'A': 10, 'B': 12, 'C': 14, 'D': 38, 'E': 20}
        self.num_zones = d_num_zones[self.store_id]
        self.x_zone_indices = tf.placeholder(tf.int32, [None, self.num_zones], name="x_zone_indices")
        self.y_RV_bin = tf.placeholder(tf.int32, [None, 2], name="y_RV_bin")
        self.y_RV_var = tf.placeholder(tf.float32, [None], name="y_RV_var")
        self.num_histories = FLAGS.max_num_histories

        self.emb_dim_uid = FLAGS.emb_dim_uid
        self.emb_dim_zone = FLAGS.emb_dim_zone

        # logs
        self.log_file = None
        self.log_filename = log_dir+'.txt'
        print('Created loader instance')

    def load_data(self):
        print('Load data for SurvRev')
        self.train_data = np.array([1,2,3])
        self.test_data = np.array([4,5,6])
        print(self.train_data+self.test_data)

    def generate_model(self):
        # Generate a model
        print('Generate a model')
        print('Running rate is {}'.format(FLAGS.learning_rate))
        # ToDo: embedding can be trained somewhere, and loaded later
        sensor_embeddings = tf.Variable(tf.random_normal(shape=[self.num_zones, self.emb_dim_zone], mean=0, stddev=0.1))
        x_zone_embs = tf.nn.embedding_lookup(sensor_embeddings, self.x_zone_indices)
        print('')
        # embeddings = tf.Variable
        #
        # //

    def train(self):
        print('Training...')
        self.write_log()

    def test(self):
        print('Testing...')
        self.write_log()

    def write_log(self):
        print('Log file name: {}'.format(self.log_filename))
        self.log_file = open(self.log_filename, 'a')
        self.log_file.write('{}_{}_{}\n'.format(self.input_file, self.log_dir, self.store_id))

    def run(self):
        self.load_data()
        self.generate_model()
        self.train()
        self.test()