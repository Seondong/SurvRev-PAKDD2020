import numpy as np
import keras
import keras.backend as K
from tensorflow2_0.params import FLAGS
import os
import csv
import time
from tensorflow import keras
import six
import random
import string
from data import *
from loss import *
from evaluation import *
import loss
from keras.layers import TimeDistributed, LSTM, Reshape, Conv1D, Concatenate, Dense, Embedding, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
import warnings
from utils import simple_attention, timeit
from keras.callbacks import CSVLogger, Callback
from collections import OrderedDict, Iterable
from keras_multi_head import MultiHead, MultiHeadAttention


warnings.filterwarnings("ignore", category=Warning)

"""
* Filename: survrev.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Our main model.
"""


# ToDo: Add diverse metrics to evaluate, implement our first model
class SurvRevK():
    """Class SurvRevK describes our main model developed by Keras (Survival Revisit Keras)"""
    def __init__(self, store_id, GPU_id):
        self.store_id = store_id
        self.GPU_id = GPU_id

        self.train_data = None
        self.test_data = None
        self.train_censored_data = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

        self.max_features = 5000
        self.max_len = 30

        self.history = None
        self.result = None
        self.result_train_censored = None

        self.y_true = None
        self.y_pred = None

        self.losses = {}

        self.time_start = time.time()
        self.time_end = None

        # Set an unique identifier for this experiment
        self.exp_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    def setup(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    @timeit
    def train_test(self):
        # Data generation
        data = Data(self.store_id)
        data.run()
        print(data.train_visits.head(3))
        print('Number of areas: {}'.format(len(data.area_embedding)))
        max_num_areas = np.max(data.train_visits.areas.apply(len))

        train_data_size = len(data.train_visits)
        print('Train size: {}, Test size: {}'.format(len(data.train_visits), len(data.test_visits)))

        # data.update_labels_train_censored_data()     # ==> Moved to data.run()    # ToDo: Check again
        if FLAGS.previous_visits:
            if FLAGS.dynamic_RNN:
                self.train_data = data.train_data_generator_hist_dynamic()
                self.test_data = data.test_data_generator_hist_dynamic()
                self.train_censored_data = data.train_censored_data_generator_hist_dynamic()
            else:
                self.train_data = data.train_data_generator_hist()
                self.test_data = data.test_data_generator_hist()
                self.train_censored_data = data.train_censored_data_generator_hist()
        else:
            self.train_data = data.train_data_generator()
            self.test_data = data.test_data_generator()
            self.train_censored_data = data.train_censored_data_generator()

        print(next(self.train_data)[0])
        # Define some loss functions
        myloss = loss.CustomLoss()

        # Initialize eval class + Run some baselines (Fast)
        eval = Evaluation()
        eval.traditional_survival_analysis_baseline(data)
        eval.traditional_survival_analysis_baseline_for_train_censored(data)
        eval.naive_baseline(data)
        eval.naive_baseline_for_train_censored(data)
        eval.poisson_process_baseline(data)
        eval.hawkes_process_baseline(data)
        eval.icdm_baseline(data)
        eval.icdm_baseline_for_train_censored(data)


        # import sys
        # sys.exit('dd')

        def example_loss(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.example_loss()

        def uc_loss_nll_date(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_loss_nll_date()

        def uc_loss_nll_week(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_loss_nll_week()

        def uc_loss_nll_month(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_loss_nll_week()

        def uc_loss_nll_season(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_loss_nll_season()

        def uc_loss_nll_day(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_loss_nll_day()

        def uc_loss_rmse(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_loss_rmse()

        def uc_c_loss_ce(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_c_loss_ce()

        def uc_c_loss_rank(y_true, y_pred):
            """Wrapper function to get each loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.uc_c_loss_rank()

        def combined_loss(y_true, y_pred):
            print(y_true[:, :-2])
            print(y_pred)
            """Wrapper function to get combined loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.combined_loss()

        # Generate a Model
        visit_embedding_layer = Embedding(
            input_dim=len(data.visit_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(data.visit_embedding.values()))],
            input_length=1,
            trainable=False)

        area_embedding_layer = Embedding(
            input_dim=len(data.area_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(data.area_embedding.values()))],
            input_length=max_num_areas,
            trainable=False)

        if FLAGS.previous_visits:
            print('yes we are here')

            single_input = keras.Input((1+max_num_areas+len(data.handcrafted_features),))
            print(single_input.shape)

            visit_input = Lambda(lambda x: x[:, 0:1])(single_input)
            visit_emb = visit_embedding_layer(visit_input)
            visit_output = Reshape((-1,))(visit_emb)

            areas_input = Lambda(lambda x: x[:, 1:-len(data.handcrafted_features)])(single_input)
            areas_emb = area_embedding_layer(areas_input)
            areas_cnn = Conv1D(filters=200, kernel_size=5,
                               padding='same', activation='relu', strides=1)(areas_emb)

            if FLAGS.multi_head:
                areas_output = simple_attention(areas_cnn, areas_cnn)
            else:
                areas_output = simple_attention(areas_cnn, areas_cnn)

            # with handcrafted features
            v_a_emb_concat = Concatenate()([visit_output, areas_output])
            v_a_output = Dense(32, activation=keras.activations.softmax)(v_a_emb_concat)
            visit_features_input = Lambda(lambda x: x[:, -len(data.handcrafted_features):])(single_input)
            concat = Concatenate()([v_a_output, visit_features_input])
            concat = BatchNormalization()(concat)

            low_level_model = keras.Model(inputs=single_input, outputs=concat)
            encoder = TimeDistributed(low_level_model)

            if FLAGS.dynamic_RNN:
                multiple_inputs = keras.Input((None, 1+max_num_areas+len(data.handcrafted_features)))
            else:
                multiple_inputs = keras.Input((FLAGS.max_num_histories, 1+max_num_areas+len(data.handcrafted_features)))

            all_areas_rslt = encoder(inputs=multiple_inputs)
            all_areas_lstm = LSTM(64, return_sequences=False)(all_areas_rslt)
            logits = Dense(365, activation='softmax')(all_areas_lstm)

            self.model = keras.Model(inputs=multiple_inputs, outputs=logits)

        else:
            visit_input = keras.Input((1,))
            print(visit_input.shape)
            visit_emb = visit_embedding_layer(visit_input)
            visit_output = Reshape((-1,))(visit_emb)

            areas_input = keras.Input((max_num_areas,))
            print(areas_input.shape)
            areas_emb = area_embedding_layer(areas_input)

            areas_cnn = Conv1D(filters=32, kernel_size=5,
                               padding='same', activation='relu', strides=1)(areas_emb)
            areas_cnn = Conv1D(filters=32, kernel_size=5,
                               padding='same', activation='relu', strides=1)(areas_cnn)
            areas_output = simple_attention(areas_cnn, areas_cnn)

            # areas_output = simple_attention(areas_emb, areas_emb)

            if FLAGS.multi_head:
                # areas_output = simple_attention(areas_cnn, areas_cnn)
                # areas_output = MultiHeadAttention(head_num=32)(areas_emb)
                # simple_attention(areas_cnn, areas_cnn))
                # areas_output = Flatten(areas_output)

                # areas_cnn = Conv1D(filters=32, kernel_size=5,
                #                    padding='same', activation='relu', strides=1)(areas_emb)
                areas_cnn = MultiHeadAttention(head_num=32)(areas_emb)
                areas_output = simple_attention(areas_cnn, areas_cnn)

            else:
                areas_cnn = Conv1D(filters=32, kernel_size=5,
                                   padding='same', activation='relu', strides=1)(areas_emb)
                areas_output = simple_attention(areas_cnn, areas_cnn)


            # with handcrafted features
            v_a_emb_concat = Concatenate()([visit_output, areas_output])
            print(K.int_shape(v_a_emb_concat))
            v_a_output = Dense(32, activation=keras.activations.softmax)(v_a_emb_concat)
            visit_features_input = keras.Input((len(data.handcrafted_features),))
            concat = Concatenate()([v_a_output, visit_features_input])
            concat = BatchNormalization()(concat)
            logits = Dense(365, activation=keras.activations.softmax)(concat)

            self.model = keras.Model(inputs=[visit_input, areas_input, visit_features_input], outputs=logits)

        # Compile model
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss=combined_loss
                           # metrics=[uc_loss_nll_date, uc_loss_nll_week, uc_loss_nll_month, uc_loss_nll_season,
                           #          uc_loss_rmse, uc_c_loss_ce, uc_c_loss_rank]
                           )  # ToDo: Fix duplicated computation. How to get sub-losses from the final loss?

        print(self.model.summary())

        # Example logger
        csv_logger = CSVLogger('../results/epochresult.csv', append=True, separator=';')

        colnames_log_callback = {'exp_id': self.exp_id,
                                 'store_id': self.store_id,
                                 'all_data': FLAGS.all_data,
                                 'num_total_epoch': FLAGS.train_epochs,
                                 'max_num_histories': FLAGS.max_num_histories,
                                 'previous_visits': FLAGS.previous_visits,
                                 'dynamic_RNN': FLAGS.dynamic_RNN,
                                 'multi_head': FLAGS.multi_head,
                                 }

        if FLAGS.all_data:  # Check more often since the whole data is much big
            steps_per_epoch = train_data_size // (10 * FLAGS.batch_size)
        else:
            steps_per_epoch = train_data_size // FLAGS.batch_size

        self.history = self.model.fit_generator(
            generator=self.train_data,
            use_multiprocessing=True,
            # workers=1,
            steps_per_epoch=steps_per_epoch,
            epochs=FLAGS.train_epochs,
            callbacks=[csv_logger, CustomLogger(data, self.test_data, self.train_censored_data, eval, colnames_log_callback)]
        )

        # Predict & Evaluate: Test data and Censored Train Data

        if FLAGS.dynamic_RNN:
            # For dynamic-rnn case, we had to yield one sample at a time to equalize the shape of the data,
            # Thus, step size should be the size of the test data
            self.result = self.model.predict_generator(
                generator=self.test_data,
                steps=len(data.test_visits)
            )
            self.result_train_censored = self.model.predict_generator(
                generator=self.train_censored_data,
                steps=len(data.train_censored_visits)
            )
        else:
            # In all other cases, generators yield test_data at once, so the step size is 1.
            self.result = self.model.predict_generator(
                generator=self.test_data,
                use_multiprocessing=True,
                # workers=1,
                steps=1
            )
            self.result_train_censored = self.model.predict_generator(
                generator=self.train_censored_data,
                use_multiprocessing=True,
                # workers=1,
                steps=1
            )

        eval.evaluate(data, self.result)
        eval.evaluate_train_censored(data, self.result_train_censored)

        self.time_end = time.time()

        # Save some infos for logging
        colnames_log_all = {'exp_id': self.exp_id,
                            'store_id': self.store_id,
                            'all_data': FLAGS.all_data,
                            'max_num_histories': FLAGS.max_num_histories,
                            'previous_visits': FLAGS.previous_visits,
                            'dynamic_RNN': FLAGS.dynamic_RNN,
                            'multi_head': FLAGS.multi_head,
                            'num_total_epoch': FLAGS.train_epochs,
                            'train_size': len(data.train_visits),
                            'test_size': len(data.test_visits),
                            'training_length': FLAGS.training_length,
                            'train_censored_size': len(data.train_censored_visits),
                            'train_revisit_ratio' : utils.get_stats(data.train_labels)['revisit_ratio'],
                            'test_revisit_ratio': utils.get_stats(data.test_labels)['revisit_ratio'],
                            'train_censored_revisit_ratio': utils.get_stats(data.train_censored_actual_labels)['revisit_ratio'],
                            'c_nll_date': myloss.c_nll_date,
                            'c_nll_week': myloss.c_nll_week,
                            'c_nll_month': myloss.c_nll_month,
                            'c_nll_season': myloss.c_nll_season,
                            'c_rmse': myloss.c_rmse,
                            'c_ce': myloss.c_ce,
                            'c_rank': myloss.c_rank,
                            'time_start': self.time_start,
                            'time_end': self.time_end,
                            'time_run': self.time_end-self.time_start,
                            }

        # Print results to log file
        eval.print_output(additional_inputs_all=colnames_log_all)

    def run(self):
        self.setup()
        self.train_test()
        print('expID: {}'.format(self.exp_id))


class CustomLogger(Callback):
    def __init__(self, data, test_data, train_censored_data, eval, colnames_log_callback, logs={}):
        self.keys = None
        self.logs = None
        self.row_dict = None
        self.data = data
        self.test_data = test_data
        self.train_censored_data = train_censored_data
        self.eval = eval
        self.colnames_log_callback = colnames_log_callback
        self.epoch_time_start = None

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):

        # Track several loss - code from keras.callback.CSVLogger
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            self.logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        self.row_dict = OrderedDict({'epoch': epoch + 1})
        self.row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.row_dict.update({'time_epoch': time.time() - self.epoch_time_start})
        self.colnames_log_callback.update(self.row_dict)

        # Predict test data and save some metrics
        if FLAGS.dynamic_RNN:
            # If-else statement is applied due to the same reason. Refer to comments in main predict part.
            result_test_current_epoch = self.model.predict_generator(generator=self.test_data, steps=len(self.data.test_visits))
            result_train_censored_current_epoch = self.model.predict_generator(generator=self.train_censored_data, steps=len(self.data.train_censored_visits))
        else:
            result_test_current_epoch = self.model.predict_generator(generator=self.test_data, steps=1)
            result_train_censored_current_epoch = self.model.predict_generator(generator=self.train_censored_data, steps=1)
        self.eval.callback_test_evaluate(self.data, result_test_current_epoch) # callback for tracking test results for each epoch.
        self.eval.callback_train_censored_evaluate(self.data, result_train_censored_current_epoch) # callback for tracking train_censored results for each epoch.
        self.eval.print_output_callback(additional_inputs_callback=self.colnames_log_callback)
        # print('callback test loss: {}'.format(K.eval(K.sum(result_current_epoch)))) # fast check this callback working
