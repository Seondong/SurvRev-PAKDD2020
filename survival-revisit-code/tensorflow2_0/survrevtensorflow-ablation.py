import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from params import FLAGS
import os
import csv
import time
import six
import random
import string
from data import *
from loss import *
from evaluation import *
import loss
from tensorflow.keras import Model
from tensorflow.keras.layers import Masking, Input, TimeDistributed, Dropout, LSTM, GRU, Reshape, Conv1D, Concatenate, Dense, Embedding, Lambda, Flatten, BatchNormalization, subtract, Bidirectional, GlobalAveragePooling1D
import warnings
from utils import simple_attention, timeit
from tensorflow.keras.callbacks import CSVLogger, Callback
from collections import OrderedDict, Iterable
# from keras_multi_head import MultiHead, MultiHeadAttention

warnings.filterwarnings("ignore", category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # No INFO, WARNING, and ERROR messages are not printed

"""
* Filename: survrevtensorflow-ablation.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Variations of the main model for ablation study.
"""


class lowLevelEncoder(Model):
    def __init__(self, data):
        super(lowLevelEncoder, self).__init__()
        self.data = data
        self.max_num_areas = np.max(self.data.train_visits.apply(len))
        # Define some embedding layers
        self.user_embedding_layer = Embedding(
            input_dim=len(self.data.visit_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(self.data.visit_embedding.values()))],
            input_length=1,
            trainable=False)
        self.area_embedding_layer = Embedding(
            input_dim=len(data.area_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(data.area_embedding.values()))],
            input_length=self.max_num_areas,
            trainable=False)

    def call(self, single_input):
        # single_input = Input((1 + self.max_num_areas + len(self.data.handcrafted_features),))
        user_input = Lambda(lambda x: x[:, 0:1])(single_input)
        user_emb = self.user_embedding_layer(user_input)
        user_output = Reshape((-1,))(user_input)

        areas_input = Lambda(lambda x: x[:, 1:-len(self.data.handcrafted_features)])(single_input)
        areas_emb = self.area_embedding_layer(areas_input)
        areas_emb = Dropout(0.1)(areas_emb)
        # areas_emb = Masking(mask_value=self.data.area_embedding['pad'], input_shape=(self.max_num_areas, FLAGS.embedding_dim))(areas_emb)

        if FLAGS.switch_low_bilstm == True:
            mask = Masking(mask_value=self.data.area_embedding['pad'],
                           input_shape=(self.max_num_areas, FLAGS.embedding_dim)).compute_mask(areas_emb)
            areas_emb = Bidirectional(LSTM(units=FLAGS.emb_dim_uid, return_sequences=True))(areas_emb, mask=mask)
        else:
            areas_emb = areas_emb

        if FLAGS.switch_low_cnn == True:
            areas_cnn = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', strides=1)(areas_emb)
        else:
            areas_cnn = areas_emb


        if FLAGS.switch_low_att == True:
            areas_output = simple_attention(areas_cnn, areas_cnn)
        else:
            areas_output = GlobalAveragePooling1D(data_format='channels_last')(areas_cnn)

        user_output = tf.cast(user_output, 'float32')
        areas_output = tf.cast(areas_output, 'float32')

        if FLAGS.switch_low_userid == True:
            v_a_emb_concat = Concatenate()([user_output, areas_output])
        else:
            v_a_emb_concat = areas_output

        v_a_output = Dense(128, activation='relu')(v_a_emb_concat)
        visit_features_input = Lambda(lambda x: x[:, -len(self.data.handcrafted_features):])(single_input)
        visit_features_input = tf.cast(visit_features_input, 'float32')

        if FLAGS.switch_low_hand == True:
            concat = Concatenate()([v_a_output, visit_features_input])
        else:
            concat = v_a_output

        concat = BatchNormalization()(concat)
        return visit_features_input


class SurvRevModel(Model):
    def __init__(self, data):
        super(SurvRevModel, self).__init__()
        self.data = data
        self.max_num_areas = np.max(self.data.train_visits.areas.apply(len))

    def call(self, multiple_inputs):
        """SurvRev model"""
        # multiple_inputs = Input((None, 1 + self.max_num_areas + len(self.data.handcrafted_features)))

        low_level_model = lowLevelEncoder(data=self.data)
        # Rather than using timedistributed which does not work currently.

        rslts = []
        for i in range(multiple_inputs.shape[1]):
            single_input = multiple_inputs[:, i, :]
            low_level_rslt = low_level_model(single_input)
            rslts.append(low_level_rslt)

        if FLAGS.switch_high_lstm == True:
            all_areas_rslt = tf.stack(rslts, axis=1)
            # all_areas_rslt = TimeDistributed(low_level_model)(multiple_inputs)
            all_areas_rslt = LSTM(256, return_sequences=True, activation='tanh')(all_areas_rslt)
            # all_areas_rslt = GRU(256, return_sequences=True, activation='relu')(all_areas_rslt)
            all_areas_rslt = LSTM(256, return_sequences=False, activation='tanh')(all_areas_rslt)

        else:
            if len(rslts) != 1:
                all_areas_rslt = Concatenate(axis=1)([x for x in rslts])
            else:
                all_areas_rslt = rslts[0]
            all_areas_rslt = Dense(256, activation='tanh')(all_areas_rslt)

        logits = Dense(365, activation='relu')(all_areas_rslt)
        logits = Dense(365, activation='relu')(logits)
        # print(logits)
        # logits = tf.multiply(logits, 1/2)
        return logits


class SurvRev():
    """Class SurvRev describes our main model"""
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

        self.colnames_log_callback = {'exp_id': self.exp_id,
                                 'store_id': self.store_id,
                                 'all_data': FLAGS.all_data,
                                 'num_total_epoch': FLAGS.train_epochs,
                                 'max_num_histories': FLAGS.max_num_histories,
                                 'previous_visits': FLAGS.previous_visits,
                                 'multi_head': FLAGS.multi_head,
                                 }


    def setup(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)

    @timeit
    def train_test(self):
        """Using training/testing set & Generated model, do learning and prediction"""

        # Data generation
        self.data = Data(self.store_id)
        self.data.run()

        print(self.data.df_train.head(3))
        print('Number of areas: {}'.format(len(self.data.area_embedding)))
        max_num_areas = np.max(self.data.train_visits.areas.apply(len))

        train_data_size = len(self.data.train_visits)
        test_data_size = len(self.data.test_visits)
        train_censored_data_size = len(self.data.train_censored_visits)
        print('Train size: {}, Test size: {}'.format(len(self.data.train_visits), len(self.data.test_visits)))

        if FLAGS.previous_visits:
            self.train_data = self.data.train_data_generator_hist_dynamic()
            self.test_data = self.data.test_data_generator_hist_dynamic()
            self.train_censored_data = self.data.train_censored_data_generator_hist_dynamic()
        else:
            self.train_data = self.data.train_data_generator()
            self.test_data = self.data.test_data_generator()
            self.train_censored_data = self.data.train_censored_data_generator()

        # Define some loss functions
        myloss = loss.CustomLoss()

        eval = Evaluation()
        # Run other baselines for performance comparison
        # eval.traditional_survival_analysis_baseline(self.data)
        # eval.traditional_survival_analysis_baseline_for_train_censored(self.data)
        # eval.naive_baseline(self.data)
        # eval.naive_baseline_for_train_censored(self.data)
        eval.poisson_process_baseline(self.data)
        # eval.hawkes_process_baseline(self.data)
        # eval.icdm_baseline(self.data)
        # eval.icdm_baseline_for_train_censored(self.data)
        # eval.WSDM17_baseline(self.data)
        # eval.AAAI19_baseline(self.data)


        print('Baselines are working!')

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
            """Wrapper function to get combined loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.combined_loss()


        self.model = SurvRevModel(data=self.data)

        # import code
        # code.interact(local=locals())
        # self.model.compile(optimizer='adam', loss=combined_loss)
        # self.model.fit(x=next(self.train_data)[0],
        #                y=next(self.train_data)[1],
        #                epochs=1,
        #                verbose=1)
        # self.model.fit_generator(generator=self.train_data,
        #                          steps_per_epoch=10,
        #                          epochs=1,
        #                          verbose=1)




        optimizer = tf.keras.optimizers.Adam()

        if FLAGS.all_data == 50000:  # Check more often since the whole data is much big
            steps_per_epoch = train_data_size // (10 * FLAGS.batch_size)
        else:
            steps_per_epoch = train_data_size // (1 * FLAGS.batch_size)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_loss1 = tf.keras.metrics.Mean(name='train_loss1')
        train_loss2 = tf.keras.metrics.Mean(name='train_loss2')
        train_loss3 = tf.keras.metrics.Mean(name='train_loss3')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
        test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
        test_loss3 = tf.keras.metrics.Mean(name='test_loss3')


        def train_step(input, label):
            with tf.GradientTape() as tape:
                predictions = self.model(input)
                loss1 = uc_c_loss_rank(label, predictions)
                loss2 = uc_loss_rmse(label, predictions)
                loss3 = uc_c_loss_ce(label, predictions)
                loss = combined_loss(label, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            train_loss(loss)
            train_loss1(loss1)
            train_loss2(loss2)
            train_loss3(loss3)

        def test_step(input, label):
            predictions = self.model(input)
            loss1 = uc_c_loss_rank(label, predictions)
            loss2 = uc_loss_rmse(label, predictions)
            loss3 = uc_c_loss_ce(label, predictions)
            loss = loss1*loss2*loss3
            test_loss(loss)
            test_loss1(loss1)
            test_loss2(loss2)
            test_loss3(loss3)

        for epoch in range(FLAGS.train_epochs):
            step = 0
            onGoing = True
            while onGoing:
                step += 1
                if step < steps_per_epoch:
                    onGoing = True
                    inputs, labels = next(self.train_data)
                    inputs = tf.cast(inputs, tf.float32)
                    labels = tf.cast(labels, tf.float32)
                    train_step(inputs, labels)
                else:
                    onGoing = False
                    test_inputs, test_labels = next(self.test_data)
                    test_inputs = tf.cast(test_inputs, tf.float32)
                    test_labels = tf.cast(test_labels, tf.float32)
                    test_step(test_inputs, test_labels)


            template = 'Epoch {}, Train-loss: {:.4f}, Rank: {:.2f}, RMSE: {:.2f}, CE: {:.2f} Test-loss: {:.2f}, TRank: {:.2f}, TRMSE: {:.2f}, TCE: {:.2f}'
            r1 = train_loss.result().numpy()
            r2 = train_loss1.result().numpy()
            r3 = train_loss2.result().numpy()
            r4 = train_loss3.result().numpy()
            r5 = test_loss.result().numpy()
            r6 = test_loss1.result().numpy()
            r7 = test_loss2.result().numpy()
            r8 = train_loss3.result().numpy()
            print(template.format(epoch + 1, r1, r2, r3, r4, r5, r6, r7, r8))

            list_keys = list(self.colnames_log_callback.keys())+\
                        ['epoch', 'train-loss', 'train-rank-loss', 'train-RMSE-loss', 'train-CE-loss', 'test-loss', 'test-rank-loss', 'test-RMSE-loss', 'test-CE-loss']

            list_results = list(self.colnames_log_callback.values())+ [epoch+1, r1, r2, r3, r4, r5, r6, r7, r8]

            # Logging
            self.callback_result_file_path = '{}{}.csv'.format(FLAGS.callback_result_dir_path, self.exp_id)
            if not os.path.exists(FLAGS.callback_result_dir_path):
                os.makedirs(FLAGS.callback_result_dir_path)
            self.exists = os.path.isfile(self.callback_result_file_path)
            with open(self.callback_result_file_path, 'a') as ff:
                wr_all = csv.writer(ff, dialect='excel')
                if self.exists:
                    wr_all.writerow(list_results)
                else:
                    wr_all.writerow(list_keys)
                    wr_all.writerow(list_results)

        test_inputs, test_labels = next(self.test_data)
        test_inputs = tf.cast(test_inputs, tf.float32)
        test_labels = tf.cast(test_labels, tf.float32)
        pred_test = self.model(test_inputs)

        print('pred_test.shape: ', pred_test.shape)

        if FLAGS.previous_visits == False:
            train_censored_inputs, train_censored_labels = next(self.train_censored_data)
            train_censored_inputs = tf.cast(train_censored_inputs, tf.float32)
            train_censored_labels = tf.cast(train_censored_labels, tf.float32)
            pred_train_censored = self.model(train_censored_inputs)
        else:
            sub_results = []
            for _ in range(len(set(self.data.df_train_censored.nvisits))):
                # This generator generates tensor with same shape (nvisits = equal).
                # So the indices of the final output are mixed descending order by nvisits
                train_censored_inputs, train_censored_labels = next(self.train_censored_data)
                predicted_single_train_censored = self.model(train_censored_inputs)
                sub_results.append(predicted_single_train_censored)
            pred_train_censored = np.concatenate(sub_results, axis=0)

            # So we re-order the final prediction results to the original order of train_censored_data
            aindex = list(self.data.df_train_censored.index)
            bindex = list(self.data.df_train_censored.sort_values(by='nvisits').index)
            right_ind_seq_for_reordering = [bindex.index(idx) for idx in aindex]
            pred_train_censored = pred_train_censored[right_ind_seq_for_reordering]

        print('pred_train_censored.shape: ', pred_train_censored.shape)

        # Evaluate the WSDM method
        eval.evaluate(self.data, pred_test, algo='SurvRev')
        eval.evaluate_train_censored(self.data, pred_train_censored, algo='SurvRev')
        print("The results of SurvRev model are listed as \"Our Model\" from the above log.")
        # import code
        # code.interact(local=locals())

        # pred_test
        # pred_train_censored


        # print(self.model.summary())
        #
        # Example logger
        csv_logger = CSVLogger('../../results/epochresult.csv', append=True, separator=';')
        #
        # colnames_log_callback = {'exp_id': self.exp_id,
        #                          'store_id': self.store_id,
        #                          'all_data': FLAGS.all_data,
        #                          'num_total_epoch': FLAGS.train_epochs,
        #                          'max_num_histories': FLAGS.max_num_histories,
        #                          'previous_visits': FLAGS.previous_visits,
        #                          'dynamic_RNN': FLAGS.dynamic_RNN,
        #                          'multi_head': FLAGS.multi_head,
        #                          }
        #


        self.time_end = time.time()
        #
        # Save some infos for logging
        colnames_log_all = {'exp_id': self.exp_id,
                            'store_id': self.store_id,
                            'all_data': FLAGS.all_data,
                            'max_num_histories': FLAGS.max_num_histories,
                            'previous_visits': FLAGS.previous_visits,
                            'multi_head': FLAGS.multi_head,
                            'num_total_epoch': FLAGS.train_epochs,
                            'train_size': len(self.data.train_visits),
                            'test_size': len(self.data.test_visits),
                            'training_length': FLAGS.training_length,
                            'train_censored_size': len(self.data.train_censored_visits),
                            'train_revisit_ratio' : utils.get_stats(self.data.df_train)['revisit_ratio'],
                            'test_revisit_ratio': utils.get_stats(self.data.df_test)['revisit_ratio'],
                            'train_censored_revisit_ratio': utils.get_stats(self.data.df_train_censored)['revisit_ratio'],
                            'c_nll_date': myloss.c_nll_date,
                            'c_nll_week': myloss.c_nll_week,
                            'c_nll_month': myloss.c_nll_month,
                            'c_nll_season': myloss.c_nll_season,
                            'c_rmse': myloss.c_rmse,
                            'c_ce': myloss.c_ce,
                            'c_rank': myloss.c_rank,
                            'switch_low_bilstm': FLAGS.switch_low_bilstm,
                            'switch_low_cnn': FLAGS.switch_low_cnn,
                            'switch_low_att': FLAGS.switch_low_att,
                            'switch_low_userid': FLAGS.switch_low_userid,
                            'switch_low_hand': FLAGS.switch_low_hand,
                            'switch_high_lstm': FLAGS.switch_high_lstm,
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
        if FLAGS.previous_visits:
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


if __name__ == "__main__":
    print("-----------------------------------------")
    print("      Running SurvRev code directly      ")
    print("-----------------------------------------")
    # gpu_id = input("Choose one GPU slot to run (ex. 0, 1, 2, 3, 4, 5, 6, 7 for DGX server)")
    gpu_id = str(FLAGS.gpu_id)
    survrev = SurvRev(store_id=FLAGS.store_id, GPU_id=gpu_id)
    survrev.run()