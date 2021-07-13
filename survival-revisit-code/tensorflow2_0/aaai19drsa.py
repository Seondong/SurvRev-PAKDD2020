import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import functools

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, TimeDistributed, LSTM, GRU, Reshape, Conv1D, Concatenate, Dense, Embedding, Lambda, Flatten, BatchNormalization
# from keras.callbacks import CSVLogger, Callback

from data import Data
from evaluation import *
from tensorflow2_0.params import FLAGS

"""
* Filename: aaai19drsa.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr) 
* Summary: Implementation of the State-of-the-art method DRSA [AAAI'19]. 
           This performance of this baseline should be comparable to our method on first-time visitor testing set, 
           whereas the performance on train_censored test set should be worse than our method. 
           We implemented it again from the scratch, to guarantee same training input and same evaluation technique. 
* Reference: Deep Recurrent Survival Analysis [AAAI'19] by Kan Ren et al.  
* Dependency: 
    * data.py (for some data preprocessing)
    * evaluation.py (for evaluation)
    * params.py (for parameters)
    * Data directory: ../data/indoor/ or ../data_samples/indoor/ depending on the parameter FLAGS.all_data
* HowTo: This script can be executed independently or via main.py.

* ToDo: Everything, Remove dependency for independent open-source. 
* Issues: 
"""


class AAAI19Data(Data):
    """Data for AAAI19"""
    def __init__(self, store_id):
        super(AAAI19Data, self).__init__(store_id)

    def train_data_generator_AAAI19(self):
        """ Train data generator for AAAI'19 DRSA.
            Consider: Using each visit separately for training, histories are not considered.
        """
        def __gen__():
            while True:
                # Only retain the last visits which includes all previous visits (Retain the last visit for each customer)
                idxs = list(self.df_train.visit_id)
                # np.random.shuffle(idxs)
                df_train = self.df_train.set_index('visit_id')
                train_visits = self.train_visits.set_index('visit_id')
                for idx in idxs:
                    visit = train_visits.loc[idx]
                    label = df_train.loc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], \
                          [label[ft] for ft in ['revisit_intention', 'suppress_time']]

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
            self.moke_data_train = np.hstack((batch[0].reshape(-1, 1), batch[1], batch[2])), batch[-1]
            yield self.moke_data_train

    def test_data_generator_AAAI19(self):
        """ Train data generator for AAAI'19 DRSA.
            Similar to train_data_generator_AAAI19()
        """
        def __gen__():
            while True:
                idxs = list(self.df_test.visit_id)
                df_all = pd.concat([self.df_train, self.df_test]).set_index('visit_id')
                visits = self.visits.set_index('visit_id')
                for idx in idxs:
                    visit = visits.loc[idx]
                    label = df_all.loc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], \
                          [label[ft] for ft in ['revisit_intention', 'suppress_time']]

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(len(self.test_visits))))]
            self.moke_data_test = np.hstack((batch[0].reshape(-1, 1), batch[1], batch[2])), batch[-1]
            yield self.moke_data_test

    def train_censored_data_generator_AAAI19(self):
        """ Train_censored data generator for AAAI'19 DRSA.
            Similar to train_data_generator_AAAI19()
        """
        def __gen__():
            while True:
                idxs = list(self.df_train_censored.visit_id)
                df_train = self.df_train.set_index('visit_id')
                train_visits = self.train_visits.set_index('visit_id')
                for idx in idxs:
                    visit = train_visits.loc[idx]
                    label = df_train.loc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], \
                          [label[ft] for ft in ['revisit_intention', 'suppress_time']]
        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(len(self.censored_visit_id))))]
            self.moke_data_train_censored = np.hstack((batch[0].reshape(-1, 1), batch[1], batch[2])), batch[-1]
            yield self.moke_data_train_censored


class AAAI19Loss():
    """Loss for AAAI19"""
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.d_interval = {'date': {'left': -0.5, 'right': 0.5},
                           'week': {'left': -3.5, 'right': 3.5},
                           'month': {'left': -15, 'right': 15},
                           'season': {'left': -45, 'right': 45}}

    def initialize_data(self, y_true, y_pred):
        """ For initializing intermediate results for calculating losses later."""
        self.y_true = y_true
        self.y_pred = y_pred

    def cal_expected_interval_for_loss(self, y_pred):
        """ For calculating MSE loss for censored data, we have to get the predict revisit interval."""
        zeros = K.zeros(shape=(FLAGS.batch_size, 1))
        tmp_concat = K.concatenate([zeros, y_pred], axis=1)
        surv_rate = K.ones_like(tmp_concat) - tmp_concat
        surv_prob = K.cumprod(surv_rate, axis=1)
        revisit_prob = surv_prob[:, :-1] * y_pred
        rint = K.variable(np.stack([np.array(range(365)) + 0.5 for _ in range(FLAGS.batch_size)], axis=0))
        pred_revisit_probability = K.sum(revisit_prob, axis=1)
        pred_revisit_interval = K.sum(rint * revisit_prob, axis=1)
        return pred_revisit_interval

    @staticmethod
    def calculate_proba(x, interval):
        """ For calculating negative log likelihood losses for censored data."""
        rvbin_label = x[-2]  # revisit binary label
        supp_time = x[-1]  # revisit suppress time  # supp_time = K.cast(K.round(supp_time), dtype='int32')
        kvar_ones = K.ones_like(x[:-2])
        y = keras.layers.Subtract()([kvar_ones, x[:-2]])  # y = non-revisit rate (1-hazard rate)

        left_bin = K.maximum(supp_time + interval['left'], K.ones_like(
            supp_time))  # reason: y[0:x] cannot be calculated when x < 1, therefore set x as 1 so that y[0:1] = 1
        right_bin = K.minimum(supp_time + interval['right'], K.ones_like(
            supp_time) * 365)  # reason: y[0:x] cannot be calculated when x > 365

        left_bin = K.cast(K.round(left_bin), dtype='int32')
        right_bin = K.cast(K.round(right_bin), dtype='int32')
        supp_time_int = K.cast(K.round(supp_time), dtype='int32')

        p_survive_until_linterval = K.prod(y[0:left_bin])  # The instance has to survive for every time step until t
        p_survive_until_rinterval = K.prod(y[0:right_bin])
        p_survive_until_supp_time = K.prod(y[0:supp_time_int])

        result = K.stack(
            [p_survive_until_linterval, p_survive_until_rinterval, p_survive_until_supp_time, rvbin_label])
        return result

    def uc_loss_nll(self, uc_loss_nll_option='date'):
        """Wrapper function for all negative log-likelihood loss"""
        probs_survive = K.map_fn(functools.partial(self.calculate_proba, interval=self.d_interval[uc_loss_nll_option]),
                                 elems=self.tmp_tensor)
        prob_revisit_at_z = tf.transpose(probs_survive)[0] - tf.transpose(probs_survive)[1]
        # If censored -> multiply by 0 -> thus ignored
        prob_revisit_at_z_uncensored = tf.add(tf.multiply(prob_revisit_at_z, tf.transpose(probs_survive)[-1]), 1e-20)
        return -tf.reduce_sum(K.log(prob_revisit_at_z_uncensored))   # / num_revisitors_in_batch

    def uc_c_loss_ce(self):
        """ Cross entropy loss (Both uncensored and censored data) """
        probs_survive = K.map_fn(functools.partial(self.calculate_proba, interval=self.d_interval['date']),
                                 elems=self.tmp_tensor, name='survive_rates')
        final_survive_prob = tf.transpose(probs_survive)[2]
        final_revisit_prob = tf.subtract(tf.constant(1.0, dtype=tf.float32), final_survive_prob)
        survive_revisit_prob = tf.transpose(tf.stack([final_survive_prob, final_revisit_prob]), name="predict")

        actual_survive_bin = tf.subtract(tf.constant(1.0, dtype=tf.float32), self.y_true[:, -2])
        actual_revisit_bin = self.y_true[:, -2]
        revisit_binary_categorical = tf.transpose(tf.stack([actual_survive_bin, actual_revisit_bin]))

        return -tf.reduce_sum(
            revisit_binary_categorical * tf.log(tf.clip_by_value(survive_revisit_prob, 1e-10, 1.0)))

    def aaai19_loss(self):
        """AAAI'19 loss, written as L_z, L_censored, L_uncensored in the paper"""
        self.tmp_tensor = K.concatenate([self.y_pred, self.y_true], axis=-1)
        print(self.y_true)
        print(self.y_pred)

        # First Loss (L_z): Negative Log Likelihood of true event time over the uncensored logs
        loss_z = self.uc_loss_nll(uc_loss_nll_option='date')

        # Second Loss (L_censored, L_uncensored): Cross Entropy
        loss_ce = self.uc_c_loss_ce()

        loss = loss_z + loss_ce
        # loss = loss_ce
        # loss = K.sum(K.mean(y_pred, axis=-1) - y_true[:, -1], axis=-1)
        return loss

    def rmse_loss(self):
        """Implementation of RMSE loss for additional testing"""
        pred_revisit_interval = self.cal_expected_interval_for_loss(self.y_pred)
        squared_error_all = K.square(keras.layers.Subtract()([pred_revisit_interval, self.y_true[:, -1]]))
        squared_error_uc = tf.multiply(squared_error_all, self.y_true[:, -2])  # y_true[-2] is a binary revisit label.
        num_revisitors_in_batch = K.sum(self.y_true[:, -2])
        loss =  K.sqrt(K.sum(squared_error_uc) / num_revisitors_in_batch)
        return loss

class AAAI19():
    def __init__(self, store_id, GPU_id):
        self.store_id = store_id
        self.GPU_id = GPU_id
        self.data = None

    def setup(self):
        pass
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        # K.set_session(sess)

    def generate_LSTM_model(self):
        """Preliminary implementation of AAAI19 model, only having LSTM layer """
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
        # concat = BatchNormalization()(concat)
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
        """AAAI19 model - Same implementation with DRSA Github repo.
           Covariate 'x' are copied for each cell and used as inputs for LSTM layer."""

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

        # Data feeding
        single_input = keras.Input((1 + self.max_num_areas + len(self.data.handcrafted_features),))
        user_input = Lambda(lambda x: x[:, 0:1])(single_input)
        area_input = Lambda(lambda x: x[:, 1:-len(self.data.handcrafted_features)])(single_input)
        visit_features_input = Lambda(lambda x: x[:, -len(self.data.handcrafted_features):])(single_input)

        user_input = user_embedding_layer(user_input)
        area_input = area_embedding_layer(area_input)  # Dimension becomes too large?

        user_input = Reshape((-1,))(user_input)
        area_input = Reshape((-1,))(area_input)

        print(user_input)
        print(area_input)
        print(visit_features_input)

        concat = Concatenate()([user_input, area_input, visit_features_input]) # [u;a;v]
        # concat = Dense(20, activation=keras.activations.softmax)(concat)
        # concat = BatchNormalization()(concat)

        print(concat.shape)
        expinp = Lambda(lambda x: K.repeat(x, 365))(concat)

        class AddTimeAscInputLayer(Layer):
            """These set of computation did not work without defining this layer"""
            def __init__(self, **kwargs):
                super(AddTimeAscInputLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                super(AddTimeAscInputLayer, self).build(input_shape)

            def call(self, x):
                ones = K.ones_like(x[:, :1, :1])
                yseq = K.variable(np.expand_dims(np.array(range(365)) + 0.5, 0))
                yseq = K.dot(ones, yseq)
                yseqd = Lambda(lambda y: K.permute_dimensions(y, (0, 2, 1)))(yseq)
                concat2 = Concatenate()([x, yseqd])
                return concat2

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1], input_shape[2]+1)

        concat2 = AddTimeAscInputLayer()(expinp)

        # LSTM
        all_areas_lstm = GRU(64, return_sequences=True)(concat2)
        logits = Dense(1, activation='softmax')(all_areas_lstm)
        logits = Lambda(lambda x: K.squeeze(x, axis=-1))(logits)
        self.model = keras.Model(inputs=single_input, outputs=logits)
        print(self.model.summary())

    def train_test(self):
        """Using training/testing set & generated model, do learning & prediction"""

        # Data generation
        self.data = AAAI19Data(self.store_id)
        self.data.run()

        print('Number of areas: {}'.format(len(self.data.area_embedding)))
        self.max_num_areas = np.max(self.data.train_visits.areas.apply(len))

        # Generate a model
        self.generate_model()

        train_data_size = len(self.data.train_visits)
        test_data_size = len(self.data.test_visits)
        train_censored_data_size = len(self.data.train_censored_visits)

        self.train_data = self.data.train_data_generator_AAAI19()
        self.test_data = self.data.test_data_generator_AAAI19()
        self.train_censored_data = self.data.train_censored_data_generator_AAAI19()

        # import code
        # code.interact(local=locals())

        myloss = AAAI19Loss()

        def aaai19_loss(y_true, y_pred):
            """Wrapper function to get combined loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.aaai19_loss()

        # def aaai19_loss2():
        #
        #     def losses(y_true, y_pred):
        #         a = 0.2
        #         loss1 = rmse_loss(y_true, preprocess1(y_pred))
        #         loss2 = ce_loss(y_true,  preprocess2(y_pred))
        #         loss = 0.2*loss1 + 0.8*loss2
        #         return loss
        #
        #     def aaai19_lossds(self):
        #         """AAAI'19 loss, written as L_z, L_censored, L_uncensored in the paper"""
        #         self.tmp_tensor = K.concatenate([self.y_pred, self.y_true], axis=-1)
        #         print(self.y_true)
        #         print(self.y_pred)
        #
        #         # First Loss (L_z): Negative Log Likelihood of true event time over the uncensored logs
        #         loss_z = self.uc_loss_nll(uc_loss_nll_option='date')
        #
        #         # Second Loss (L_censored, L_uncensored): Cross Entropy
        #         loss_ce = self.uc_c_loss_ce()
        #         print(K.eval(loss_ce))
        #
        #         # loss = loss_z + loss_ce
        #         loss = loss_ce
        #         # loss = K.sum(K.mean(y_pred, axis=-1) - y_true[:, -1], axis=-1)
        #         return loss

        def rmse_loss(y_true, y_pred):
            """Wrapper function to get combined loss by feeding data"""
            myloss.initialize_data(y_true, y_pred)
            return myloss.rmse_loss()


        # Compile model
        self.model.compile(optimizer=keras.optimizers.Adam(0.01),
                           loss=aaai19_loss    # rmse_loss
                           )

        if FLAGS.all_data:  # Check more often since the whole data is much big
            steps_per_epoch = train_data_size // (10 * FLAGS.batch_size)
        else:
            steps_per_epoch = train_data_size // FLAGS.batch_size

        self.history = self.model.fit_generator(
            generator=self.train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=FLAGS.train_epochs,
        )

        self.result = self.model.predict_generator(
            generator=self.test_data,
            steps=1
        )

        print(self.result)

        self.result_train_censored = self.model.predict_generator(
            generator=self.train_censored_data,
            steps=1
        )

        print(self.result_train_censored)

        eval = Evaluation()

        # Run other baselines for performance comparison
        eval.naive_baseline(self.data)
        eval.naive_baseline_for_train_censored(self.data)
        eval.poisson_process_baseline(self.data)
        eval.hawkes_process_baseline(self.data)
        eval.icdm_baseline(self.data)
        eval.icdm_baseline_for_train_censored(self.data)
        eval.traditional_survival_analysis_baseline(self.data)
        eval.traditional_survival_analysis_baseline_for_train_censored(self.data)

        # Evaluate the AAAI19 method
        eval.evaluate(self.data, self.result)
        eval.evaluate_train_censored(self.data, self.result_train_censored)
        print("The results of AAAI'19 model are listed as \"Our Model\" from the above log.")

    def run(self):
        self.setup()
        self.train_test()


if __name__ == "__main__":
    print("-----------------------------------------")
    print("      Running AAAI'19 code directly      ")
    print("-----------------------------------------")
    # gpu_id = input("Choose one GPU slot to run (ex. 0, 1, 2, 3, 4, 5, 6, 7 for DGX server)")
    gpu_id = "0"
    aaai19 = AAAI19(store_id=FLAGS.store_id, GPU_id=gpu_id)
    aaai19.run()
