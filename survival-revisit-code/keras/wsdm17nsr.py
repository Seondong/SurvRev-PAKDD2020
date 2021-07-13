import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import functools

import keras
import keras.backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Layer, TimeDistributed, LSTM, Reshape, Conv1D, Concatenate, Dense, Embedding, Lambda, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger, Callback

from data import Data
from evaluation import *
from tensorflow2_0.params import FLAGS

"""
* Filename: wsdm17nsr.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr) 
* Summary: Implementation of the State-of-the-art method NSR [WSDM'17]. 
           This performance of this baseline should be comparable to our method on train_censored set, whereas the 
           performance on first-visitor test set should be worse than our method. Currently, this conjecture is true
           by using rmse_loss on self.model.compile(), but it does not hold by using wsdm_loss. 
           We implemented it from the scratch, since the code was not opened by the authors of WSDM'17.
* Reference: Neural Survival Recommender [WSDM'17] by How Jing et al.  
* Dependency: 
    * data.py (for some data preprocessing)
    * evaluation.py (for evaluation)
    * params.py (for parameters)
    * Data directory: ../data/indoor/ or ../data_samples/indoor/ depending on the parameter FLAGS.all_data
* HowTo: This script can be executed independently or via main.py.

* ToDo: Gap embedding layer, Remove dependency for independent open-source. 
* Issues: 
      * Please check the consistency of wsdm_loss by comparing it to the original paper, 
        since it leads to worse performance than using rmse_loss.
"""

class WSDMData(Data):
    """Data for WSDM"""
    def __init__(self, store_id):
        super(WSDMData, self).__init__(store_id)

    def train_data_generator_WSDM(self):
        """ Train data generator for WSDM'17 Neural Survival Recommender.
            Consider: Previous histories, Dynamic LSTM,
                      Using only the last visit for each wifi_id, Flush labels for all visits
        """
        def __gen__():
            while True:
                # Only retain the last visits which includes all previous visits (Retain the last visit for each customer)
                idxs = list(self.df_train.drop_duplicates(subset='wifi_id', keep='last').visit_id)
                idxs = self.custom_shuffle(idxs)
                df_train = self.df_train.set_index('visit_id')
                train_visits = self.train_visits.set_index('visit_id')
                for idx in idxs:
                    prev_idxs = self.find_prev_nvisits(idx, numhistories=FLAGS.max_num_histories)
                    assert idx == prev_idxs[-1]
                    # # For fixing number of events for easier debugging purpose (breaking dynamic RNN temporarily)
                    # # To use this, we have to change the first dimension of the variable---multiple_inputs---
                    # # from None -> FLAGS.max_num_histories at the same time.
                    # prev_idxs = [None for _ in range(FLAGS.max_num_histories - len(prev_idxs))] + prev_idxs
                    output1, output2, output3, output4 = [], [], [], []
                    for pidx in prev_idxs[:FLAGS.max_num_histories]:
                        try:
                            visit = train_visits.loc[pidx]
                            label = df_train.loc[pidx]
                            output1 = np.append(output1, visit['visit_indices'])
                            output2 = np.append(output2, visit['area_indices'])
                            output3 = np.append(output3, [visit[ft] for ft in self.handcrafted_features])
                            output4 = np.append(output4, [label[ft] for ft in ['revisit_intention', 'suppress_time']])
                        except TypeError:
                            output1 = np.append(output1, [self.pad_val_visit])
                            output2 = np.append(output2, np.full(shape=self.num_area_thres, fill_value=self.pad_val_area))
                            output3 = np.append(output3, np.zeros(len(self.handcrafted_features)))
                            output4 = np.append(output4, np.array([-1.0, -1.0]))

                    yield np.hstack((output1.reshape(-1, 1), output2.reshape(-1, len(visit['area_indices'])),
                                     output3.reshape(-1, len(self.handcrafted_features)))), output4.reshape(-1, 2)  # only the last visit's labels (Last two elements)

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
            moke_data = batch[0], batch[-1]
            yield moke_data

    def test_data_generator_WSDM(self):
        """ Test data generator for WSDM'17 Neural Survival Recommender.
            Similar to train_data_generator_WSDM.
        """
        def __gen__():
            while True:
                idxs = list(self.df_test.visit_id)
                df_all = pd.concat([self.df_train, self.df_test]).set_index('visit_id')
                visits = self.visits.set_index('visit_id')
                for idx in idxs:
                    prev_idxs = self.find_prev_nvisits(idx, numhistories=FLAGS.max_num_histories)
                    assert idx == prev_idxs[-1]
                    output1, output2, output3, output4 = [], [], [], []
                    for pidx in prev_idxs[:FLAGS.max_num_histories]:
                        try:
                            visit = visits.loc[pidx]
                            label = df_all.loc[pidx]
                            output1 = np.append(output1, visit['visit_indices'])
                            output2 = np.append(output2, visit['area_indices'])
                            output3 = np.append(output3, [visit[ft] for ft in self.handcrafted_features])
                            output4 = np.append(output4, [label[ft] for ft in ['revisit_intention', 'suppress_time']])
                        except TypeError:
                            output1 = np.append(output1, [self.pad_val_visit])
                            output2 = np.append(output2, np.full(shape=self.num_area_thres, fill_value=self.pad_val_area))
                            output3 = np.append(output3, np.zeros(len(self.handcrafted_features)))
                            output4 = np.append(output4, np.array([-1, -1]))

                    yield np.hstack((output1.reshape(-1, 1), output2.reshape(-1, len(visit['area_indices'])),
                                     output3.reshape(-1, len(self.handcrafted_features)))), output4.reshape(-1, 2)  # only the last visit's labels (Last two elements)

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(1)))]
            moke_data = batch[0], batch[-1]
            self.moke_data_test = moke_data  # Save it temporarily for debugging purpose   # ToDo: Remove
            yield moke_data

    def train_censored_data_generator_WSDM(self):
        """ Train-censored data generator for WSDM'17 Neural Survival Recommender.
            Similar to train_data_generator_WSDM.
        """
        def __gen__():
            while True:
                idxs = list(self.df_train_censored.visit_id)
                df_train = self.df_train.set_index('visit_id')
                train_visits = self.train_visits.set_index('visit_id')
                for idx in idxs:
                    prev_idxs = self.find_prev_nvisits(idx, numhistories=FLAGS.max_num_histories)
                    assert idx == prev_idxs[-1]
                    output1, output2, output3, output4 = [], [], [], []
                    for pidx in prev_idxs[:FLAGS.max_num_histories]:
                        try:
                            visit = train_visits.loc[pidx]
                            label = df_train.loc[pidx]
                            output1 = np.append(output1, visit['visit_indices'])
                            output2 = np.append(output2, visit['area_indices'])
                            output3 = np.append(output3, [visit[ft] for ft in self.handcrafted_features])
                            output4 = np.append(output4, [label[ft] for ft in ['revisit_intention', 'suppress_time']])
                        except TypeError:
                            output1 = np.append(output1, [self.pad_val_visit])
                            output2 = np.append(output2, np.full(shape=self.num_area_thres, fill_value=self.pad_val_area))
                            output3 = np.append(output3, np.zeros(len(self.handcrafted_features)))
                            output4 = np.append(output4, np.array([-1, -1]))

                    yield np.hstack((output1.reshape(-1, 1), output2.reshape(-1, len(visit['area_indices'])),
                                     output3.reshape(-1, len(self.handcrafted_features)))), output4.reshape(-1, 2)  # only the last visit's labels (Last two elements)

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(1)))]
            moke_data = batch[0], batch[-1]
            self.moke_data_train_censored = moke_data  # Save it temporarily for debugging purpose
            yield moke_data

class WSDMLoss():
    """Loss for WSDM"""
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def initialize_data(self, y_true, y_pred):
        """ For initializing intermediate results for calculating losses later."""
        self.y_true = y_true
        self.y_pred = y_pred

    def cal_expected_interval_for_loss(self, y_pred):
        """ For calculating MSE loss for censored data, we have to get the predict revisit interval.
            Used only for rmse_loss"""
        zeros = K.zeros_like(y_pred[:, :, :1])
        tmp_concat = K.concatenate([zeros, y_pred], axis=-1)
        surv_rate = K.ones_like(tmp_concat) - tmp_concat
        surv_prob = K.cumprod(surv_rate, axis=-1)
        revisit_prob = surv_prob[:, :, :-1] * y_pred

        # Had a issue of unknown length dimension expansion with previous code used in survrev.py
        # Link:  https://stackoverflow.com/questions/54960227/keras-repeat-elements-with-unknown-batch-size
        # Previous code:
        # flex_dim = K.int_shape(y_pred)[1]    # => None
        # rint = K.variable(np.stack([np.stack([np.array(range(365)) + 0.5 for _ in range(flex_dim)]) for _ in
        #                    range(FLAGS.batch_size)]))

        # New solution:
        aa = K.ones_like(y_pred)[:, :, :1]
        # We used 365 days instead of 180 days (= 4320 hours in the paper) owing to the nature of longer revisitation
        # pattern of the off-line stores compared to on-line music streaming site
        # Additionally, we can reduce the parameter size and obtain fast computation."""
        avg_bin = (np.array(range(365)) + 0.5).astype('float32')
        bb = K.expand_dims(avg_bin, axis=0)
        rint = K.dot(aa, bb)

        pred_revisit_probability = K.sum(revisit_prob, axis=-1)
        pred_revisit_interval = K.sum(rint * revisit_prob, axis=-1)
        return pred_revisit_interval

    @staticmethod
    def calculate_rate_sum_until_supp_time(y):
        """ For summing up revisit rates until suppress time."""

        def _calculate_proba(x):
            supp_time = x[-1]  # revisit suppress time, which represents both b_i - e_{i-1}, T - e_{m_u}.
            supp_time_int = K.cast(K.round(supp_time-0.5), dtype='int32')  # (e.g., if supp_time = 17.7 -> round(17.7-0.5) = 17, if supp_time = 18.2 -> round(18.2-0.5) = 18)
            rate_sum_until_supp_time = K.sum(x[0:supp_time_int])  # Sum rates up to the maximum integer before suppress time.
            # print('calculate_rate_sum_until_supp_time - x shape: ', K.int_shape(x))
            # print('calculate_rate_sum_until_supp_time - supp_time shape: ', K.int_shape(supp_time))
            # print('calculate_rate_sum_until_supp_time - supp_time_int shape: ', K.int_shape(supp_time_int))
            # print('calculate_rate_sum_until_supp_time - p_survive_until_supp_time shape: ', K.int_shape(p_survive_until_supp_time))
            return rate_sum_until_supp_time

        sum_revisit_rate_before_supp_time = K.map_fn(functools.partial(_calculate_proba),
                                              elems=y)
        # print('calculate_rate_sum_until_supp_time - sum_revisit_rate_before_bi shape: ', K.int_shape(sum_revisit_rate_before_bi))
        return sum_revisit_rate_before_supp_time

    @staticmethod
    def calculate_rate_at_supp_time(y):
        """ For calculate revisit rate at suppress time."""

        def _calculate_proba(x):
            supp_time = x[-1]
            supp_time_int = K.cast(K.round(supp_time - 0.5), dtype='int32')
            rate_at_supp_time = K.sum(x[supp_time_int:supp_time_int+1])
            # print('calculate_rate_at_supp_time - rate_at_supp_time shape: ', K.int_shape(rate_at_supp_time))
            return rate_at_supp_time

        revisit_rate_at_supp_time = K.map_fn(functools.partial(_calculate_proba), elems=y)
        return revisit_rate_at_supp_time

    def wsdm_loss(self):
        """WSDM'17 loss, written as L_{rec} in the paper"""

        y_true = self.y_true  # Dim: [batch_size, 1-5, 2]
        y_pred = self.y_pred  # Dim: [batch_size, 1-5, 365]
        self.tmp_tensor = K.concatenate([y_pred, y_true], axis=-1)

        # # Minimizing objective: Negative Log Likelihood of gap probability (L_{rec} has three terms in the paper.)
        # # Revisit rate before suppress time should be minimized. (Summation of The first, the third term)
        sum_revisit_rate_before_supp_time = K.map_fn(functools.partial(self.calculate_rate_sum_until_supp_time),
                                 elems=self.tmp_tensor, name='sum_revisit_rate_before_supp_time')

        # Revisit rate at suppress time. (The second term)
        revisit_rate_at_supp_time = K.map_fn(functools.partial(self.calculate_rate_at_supp_time),
                                              elems=self.tmp_tensor, name='revisit_rate_at_supp_time')

        # ToDo Issue: How large dt value to be multipled on the first and the second term? (Is considering dt = 1 okay?)
        print('wsdm_loss - revisit_rate_at_supp_time shape: ', K.int_shape(revisit_rate_at_supp_time))

        loss = tf.reduce_sum(sum_revisit_rate_before_supp_time) - tf.reduce_sum(revisit_rate_at_supp_time)
        return loss

    def rmse_loss(self):
        """Implementation of RMSE loss for additional testing"""
        y_true = self.y_true
        y_pred = self.y_pred
        pred_revisit_interval = self.cal_expected_interval_for_loss(y_pred)
        squared_error_all = K.square(keras.layers.Subtract()([pred_revisit_interval, self.y_true[:, :, -1]]))
        squared_error_uc = squared_error_all * self.y_true[:, :, -2]  # y_true[-2] is a binary revisit label.
        loss = K.sqrt(K.sum(squared_error_uc))
        return loss


class WSDM():
    def __init__(self, store_id, GPU_id):
        self.store_id = store_id
        self.GPU_id = GPU_id
        self.data = None

    def setup(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    def generate_LSTM_model(self):
        """Preliminary implementation of WSDM model, only having LSTM layer """
        single_input = keras.Input((1 + self.max_num_areas + len(self.data.handcrafted_features),))

        # Generate a Model
        visit_embedding_layer = Embedding(
            input_dim=len(self.data.visit_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(self.data.visit_embedding.values()))],
            input_length=1,
            trainable=False)

        # x=[g;d;a;u], g=Gap,d=Time,a=Areas(Actions),u=Wifi_id
        gap_input = Lambda(lambda x: x[:, -1:])(single_input)
        time_idx = -len(self.data.handcrafted_features)+list(self.data.handcrafted_features).index('day_hour_comb')
        time_input = Lambda(lambda x: x[:, time_idx:time_idx+1])(single_input)
        area_input = Lambda(lambda x: x[:, 1:-len(self.data.handcrafted_features)])(single_input)
        visit_input = Lambda(lambda x: x[:, 0:1])(single_input)

        gap_input = Reshape((-1,))(gap_input)
        time_input = Reshape((-1,))(time_input)
        area_input = Reshape((-1,))(area_input)
        visit_input = Reshape((-1,))(visit_input)

        # with handcrafted features
        concat = Concatenate()([gap_input, time_input, area_input, visit_input])
        concat = Dense(40, activation=keras.activations.softmax)(concat)
        concat = BatchNormalization()(concat)
        print('concat: ', K.int_shape(concat))
        low_level_model = keras.Model(inputs=single_input, outputs=concat)
        encoder = TimeDistributed(low_level_model)

        if FLAGS.dynamic_RNN:
            multiple_inputs = keras.Input((None, 1 + self.max_num_areas + len(self.data.handcrafted_features)))
        else:
            multiple_inputs = keras.Input((FLAGS.max_num_histories, 1 + self.max_num_areas + len(self.data.handcrafted_features)))

        all_areas_rslt = encoder(inputs=multiple_inputs)
        print('all_areas_rslt: ', K.int_shape(all_areas_rslt))
        all_areas_lstm = LSTM(64, return_sequences=True)(all_areas_rslt)

        logits = Dense(365, activation='softmax')(all_areas_lstm)

        self.model = keras.Model(inputs=multiple_inputs, outputs=logits)
        print(self.model.summary())

    def generate_model(self):
        """WSDM model"""

        # Define some embedding layers
        user_embedding_layer = Embedding(
            input_dim=len(self.data.visit_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(self.data.visit_embedding.values()))],
            input_length=1,
            trainable=False)

        area_embedding_layer = Embedding(
            input_dim=len(self.data.area_embedding),
            output_dim=FLAGS.embedding_dim,
            weights=[np.array(list(self.data.area_embedding.values()))],
            input_length=self.max_num_areas,
            trainable=False)

        # ToDo: Gap embedding layer
        # gap_embedding_layer = Embedding(
        #     input_dim=len(self.data.area_embedding),
        #     output_dim=FLAGS.embedding_dim,
        #     weights=[np.random.rand(3,FLAGS.embedding_dim)],
        #     input_length=1,
        #     trainable=True)

        time_embedding_layer = Embedding(
            input_dim=168,
            output_dim=FLAGS.embedding_dim,
            weights=[np.random.rand(168,FLAGS.embedding_dim)],
            input_length=1,
            trainable=True)

        # LSTM part
        # x=[g;d;a;u], g=Gap,d=Time,a=Areas(Actions),u=Wifi_id
        single_input = keras.Input((1 + self.max_num_areas + len(self.data.handcrafted_features),))

        gap_input = Lambda(lambda x: x[:, -1:])(single_input)
        time_idx = -len(self.data.handcrafted_features)+list(self.data.handcrafted_features).index('day_hour_comb')
        time_input = Lambda(lambda x: x[:, time_idx:time_idx+1])(single_input)
        area_input = Lambda(lambda x: x[:, 1:-len(self.data.handcrafted_features)])(single_input)
        user_input = Lambda(lambda x: x[:, 0:1])(single_input)

        # ToDo: Turn on Embedding later
        time_input = time_embedding_layer(time_input)
        area_input = area_embedding_layer(area_input) # equivalent to action
        user_input = user_embedding_layer(user_input)  # user_input

        print('gap_input: ', K.int_shape(gap_input))
        print('time_input: ', K.int_shape(time_input))
        print('area_input: ', K.int_shape(area_input))
        print('user_input: ', K.int_shape(user_input))

        gap_input = Reshape((-1,))(gap_input)
        time_input = Reshape((-1,))(time_input)
        area_input = Reshape((-1,))(area_input)
        user_input = Reshape((-1,))(user_input)

        print('gap_input: ', K.int_shape(gap_input))
        print('time_input: ', K.int_shape(time_input))
        print('area_input: ', K.int_shape(area_input))
        print('user_input: ', K.int_shape(user_input))

        concat = Concatenate()([gap_input, time_input, area_input, user_input])  # [g;d;a;u]
        # concat = Dense(40, activation=keras.activations.softmax)(concat)
        # concat = BatchNormalization()(concat)
        print('concat: ', K.int_shape(concat))
        low_level_model = keras.Model(inputs=single_input, outputs=concat)
        encoder = TimeDistributed(low_level_model)

        multiple_inputs = keras.Input((None, 1 + self.max_num_areas + len(self.data.handcrafted_features))) # FLAGS.max_num_histories if FLAGS.dynamic_RNN:
        all_areas_rslt = encoder(inputs=multiple_inputs)
        all_areas_lstm = LSTM(1024, return_sequences=True)(all_areas_rslt)

        # Combine two inputs for inserting into 3-way factored unit
        # Instead of making input list we concatenate two inputs for using TimeDistributed wrapper.
        user_input_for_TD = Lambda(lambda x: x[:, :, 0:1])(multiple_inputs)  # tiled same user_input data for TD
        print('multipleinpout', multiple_inputs)

        # Apply embedding layer on tiled user inputs
        user_input_for_TD1 = TimeDistributed(user_embedding_layer)(user_input_for_TD)
        print('user_input_for_TD: ', user_input_for_TD1)
        print('user_input_for_TD: ', K.int_shape(user_input_for_TD1))  # Have to flatten 3rd and 4th dim into one
        user_input_for_TD2 = Lambda(lambda x: x[:, :, 0, :])(user_input_for_TD1)
        # user_input_for_TD2 = K.squeeze(user_input_for_TD1, axis=-2)  # If output gives tf.xxx then keras graph computation fails
        print('user_input_for_TD: ', user_input_for_TD2)
        print('user_input_for_TD: ', K.int_shape(user_input_for_TD2))

        aggre_threeway_inputs = Concatenate()([user_input_for_TD2, all_areas_lstm])
        threeway_encoder = TimeDistributed(ThreeWay(output_dim=512))
        three_way_rslt = threeway_encoder(aggre_threeway_inputs)
        logits = Dense(365, activation='softmax')(three_way_rslt)  # ToDo: This should be changed to TimedistributedDense

        self.model = keras.Model(inputs=multiple_inputs, outputs=logits)
        print(self.model.summary())

    def train_test(self):
        """Using training/testing set & generated model, do learning & prediction"""

        # Data generation
        self.data = WSDMData(self.store_id)
        self.data.run()

        print(self.data.train_visits.head(3))
        print('Number of areas: {}'.format(len(self.data.area_embedding)))
        self.max_num_areas = np.max(self.data.train_visits.areas.apply(len))

        # Generate a model
        self.generate_model()

        train_data_size = len(self.data.train_visits)
        test_data_size = len(self.data.test_visits)
        train_censored_data_size = len(self.data.train_censored_visits)

        myloss = WSDMLoss()
        def wsdm_loss(y_true, y_pred):
            """Wrapper function to get combined loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.wsdm_loss()

        def rmse_loss(y_true, y_pred):
            """Wrapper function to get combined loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.rmse_loss()

        if FLAGS.dynamic_RNN:     # Does not care - only use dynamic RNN case   # ToDo: Remove the other case
            self.train_data = self.data.train_data_generator_WSDM()
            self.test_data = self.data.test_data_generator_WSDM()
            self.train_censored_data = self.data.train_censored_data_generator_WSDM()
        else:
            self.train_data = self.data.train_data_generator_WSDM()
            self.test_data = self.data.test_data_generator_WSDM()
            self.train_censored_data = self.data.train_censored_data_generator_WSDM()

        # Compile model
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss=rmse_loss    # rmse_loss
                           )

        if FLAGS.all_data:  # Check more often since the whole data is much big
            steps_per_epoch = train_data_size // (10 * FLAGS.batch_size)
        else:
            steps_per_epoch = train_data_size // FLAGS.batch_size

        self.history = self.model.fit_generator(
            generator=self.train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=FLAGS.train_epochs,
            # use_multiprocessing=True,
            # workers=1,
            # callbacks=[csv_logger,
            #            CustomLogger(data, self.test_data, self.train_censored_data, eval, colnames_log_callback)]
        )

        self.result = self.model.predict_generator(
            generator=self.test_data,
            steps=test_data_size
        )
        last_dim_result = self.result[:, -1, :]

        # Previous code: As below
        # Error message: ValueError: all the input array dimensions except for the concatenation axis must match exactly
        # ToDo: Make this generator work for our case
        # self.result_train_censored = self.model.predict_generator(
        #     generator=self.train_censored_data,
        #     steps=train_censored_data_size
        # )

        sub_results = []
        for _ in range(train_censored_data_size):
            x = next(self.train_censored_data)
            predicted_single_train_censored = self.model.predict(x[0], batch_size=1)
            sub_results.append(predicted_single_train_censored[:, -1, :])

        last_dim_result_train_censored = np.concatenate(sub_results, axis=0)

        eval = Evaluation()

        # Run other baselines for performance comparison
        eval.naive_baseline(self.data)
        eval.naive_baseline_for_train_censored(self.data)
        eval.poisson_process_baseline(self.data)
        eval.hawkes_process_baseline(self.data)
        eval.icdm_baseline(self.data)
        eval.icdm_baseline_for_train_censored(self.data)
        # eval.traditional_survival_analysis_baseline(self.data)
        # eval.traditional_survival_analysis_baseline_for_train_censored(self.data)

        # Evaluate the WSDM method
        eval.evaluate(self.data, last_dim_result)
        eval.evaluate_train_censored(self.data, last_dim_result_train_censored)
        print("The results of WSDM'17 model are listed as \"Our Model\" from the above log.")

    def run(self):
        self.setup()
        self.train_test()

class ThreeWay(Layer):
    """
    Implementation of 3-way factored unit in WSDM'17 paper.

    # Arguments
        A concatenated tensor, x = [u, h]
        Tensor u: Tensor (User embedding)
        Tensor h: Tensor (LSTM output)
    # Returns
        A tensor, big_lambda (event rate)
    # Implementation Details
        Instead of getting u,h separately as a list, we aggregated those two inputs first and get those together as a
        single input, since we had to plug ThreeWay into TimeDistributed method, which only allows single input. We use
        ThreeWay in TimeDistributed to feed LSTM result which is a temporal sequence of matrices,

    # Examples
    ```python
        >>> import numpy as np
        >>> import keras.backend as K
        >>> a = K.variable(np.random.rand(32, 300))  # Consider that u=(32, 100), h=(32, 200), when size(mini_batch)=32
        >>> ThreeWay(output_dim=64)(a)
        <tf.Tensor 'three_way_1/Softplus:0' shape=(32, 64) dtype=float32>
    ```

    # References:
    1. Custom layer How-To: https://keras.io/layers/writing-your-own-keras-layers/
    2. How-To with more description: https://keunwoochoi.wordpress.com/2016/11/18/for-beginners-writing-a-custom-keras-layer/
    3. Dense function in keras.layers.core: https://github.com/keras-team/keras/blob/master/keras/layers/core.py
    4. GRU cell in keras.layers.recurrent: https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py
    5. Custom GRU layer: https://neurowhai.tistory.com/289
    6. GRU Tutorial(Eq): https://ratsgo.github.io/deep%20learning/2017/05/13/GRU/
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ThreeWay, self).__init__(**kwargs)

    def build(self, input_shape):
        """Weights are defined here.
           Some weights are commented since in our dataset, we don't do multi-task learning for another recommendation.
        """

        input_u_shape = 1
        input_h_shape = input_shape[-1] - 1

        self.kernel_w_fu = self.add_weight(name='kernel_w_fu',
                                           shape=(input_u_shape, self.output_dim),
                                           initializer='uniform',
                                           trainable=False)
        self.kernel_w_fh = self.add_weight(name='kernel_w_fh',
                                           shape=(input_h_shape, self.output_dim),
                                           initializer='uniform',
                                           trainable=False)
        # self.kernel_w_rec = self.add_weight(name='kernel_w_rec',
        #                                     shape=(input_h_shape, self.output_dim),
        #                                     initializer='uniform',
        #                                     trainable=True)
        self.kernel_w_sur = self.add_weight(name='kernel_w_sur',
                                            shape=(input_h_shape, self.output_dim),
                                            initializer='uniform',
                                            trainable=False)
        self.kernel_w_fr = self.add_weight(name='kernel_w_fr',
                                           shape=(self.output_dim, self.output_dim),
                                           initializer='uniform',
                                           trainable=False)
        self.kernel_w_fs = self.add_weight(name='kernel_w_fs',
                                           shape=(self.output_dim, self.output_dim),
                                           initializer='uniform',
                                           trainable=False)
        # self.bias_rec = self.add_weight(name='bias_rec',
        #                                 shape=(self.output_dim,),
        #                                 initializer='uniform',
        #                                 trainable=True)
        self.bias_sur = self.add_weight(name='bias_sur',
                                        shape=(self.output_dim,),
                                        initializer='uniform',
                                        trainable=False)

        super(ThreeWay, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        """3-way factored unit"""
        u = Lambda(lambda y: y[:, :1])(x)
        h = Lambda(lambda y: y[:, 1:])(x)

        f_u = K.dot(u, self.kernel_w_fu)
        f_h = K.dot(h, self.kernel_w_fh)

        ## Uncomment for additional recommendation task
        # alpha = K.dot(h, self.kernel_w_rec) + K.dot((f_u * f_h), self.kernel_w_fr) + self.bias_rec
        # alpha = keras.activations.softplus(alpha)  # softrelu activation

        big_lambda = K.dot(h, self.kernel_w_sur) + K.dot((f_u * f_h), self.kernel_w_fs) + self.bias_sur
        big_lambda = keras.activations.softplus(big_lambda)  # softrelu activation

        return big_lambda

    def compute_output_shape(self, input_shape):
        """NOTE: Must need this code block to infer output dimensions"""
        return input_shape[:-1]+(self.output_dim,)


if __name__ == "__main__":
    print("-----------------------------------------")
    print("      Running WSDM'17 code directly      ")
    print("-----------------------------------------")
    gpu_id = input("Choose one GPU slot to run (ex. 0, 1, 2, 3, 4, 5, 6, 7 for DGX server)")
    wsdm = WSDM(store_id=FLAGS.store_id, GPU_id=gpu_id)
    wsdm.run()
