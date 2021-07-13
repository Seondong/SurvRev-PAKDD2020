import os
import numpy as np
import pandas as pd
import functools
from data import Data
from evaluation import *
from params import FLAGS
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, TimeDistributed, LSTM, GRU, Reshape, Conv1D, Concatenate, Dense, Embedding, Lambda, Flatten, BatchNormalization, subtract


# from keras.backend.tensorflow_backend import set_session

# from keras.callbacks import CSVLogger, Callback

# import matplotlib
# matplotlib.use('agg')

"""
* Filename: aaai19tensorflow2.py (Tensorflow v2.0)
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

* ToDo: LSTM fix, Test evaluation add, Remove dependency for independent open-source. 
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
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
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
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
            self.moke_data_train_censored = np.hstack((batch[0].reshape(-1, 1), batch[1], batch[2])), batch[-1]
            yield self.moke_data_train_censored

class AAAI19Model(Model):
    def __init__(self, data):
        super(AAAI19Model, self).__init__()
        self.d1 = Dense(40, activation='softmax')
        self.data = data
        self.max_num_areas = np.max(self.data.train_visits.areas.apply(len))

    def call(self, single_input):
        user_input = Lambda(lambda x: x[:, 0:1])(single_input)
        area_input = Lambda(lambda x: x[:, 1:-len(self.data.handcrafted_features)])(single_input)
        visit_features_input = Lambda(lambda x: x[:, -len(self.data.handcrafted_features):])(single_input)

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

        user_input = user_embedding_layer(user_input)
        area_input = area_embedding_layer(area_input)  # Dimension becomes too large?

        user_input = Reshape((-1,))(user_input)
        area_input = Reshape((-1,))(area_input)

        concat = Concatenate()([user_input, area_input, visit_features_input])  # [u;a;v]
        concat = Dense(128, activation='relu')(concat)
        concat = BatchNormalization()(concat)

        expinp = Lambda(lambda x: K.repeat(x, 365))(concat)

        # Add time from 1-365
        ones = K.ones_like(expinp[:, :1, :1])
        yseq = K.variable(np.expand_dims(np.array(range(365)) + 0.5, 0))
        yseq = K.dot(ones, yseq)
        yseqd = Lambda(lambda y: K.permute_dimensions(y, (0, 2, 1)))(yseq)
        expinp = Concatenate()([expinp, yseqd])
        # !!!!!!
        # all_areas_lstm = tf.compat.v1.nn.static_rnn(tf.compat.v1.nn.rnn_cell.GRUCell(64), inputs=expinp)
        all_areas_lstm = LSTM(64, return_sequences=True)(expinp)
        # print("!!!!!", expinp)
        logits = TimeDistributed(Dense(1, activation='sigmoid'))(all_areas_lstm)
        logits = Lambda(lambda x: K.squeeze(x, axis=-1))(logits)
        return logits

class AAAI19():
    def __init__(self, store_id, GPU_id, for_baseline=False):
        self.store_id = store_id
        self.GPU_id = GPU_id
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_censored_data = None
        self.d_interval = {'date': {'left': -0.5, 'right': 0.5},
                           'week': {'left': -3.5, 'right': 3.5},
                           'month': {'left': -15, 'right': 15},
                           'season': {'left': -45, 'right': 45}}
        self.tmp_tensor = None
        self.probs_survive = None
        self.for_baseline= for_baseline

    def setup(self):
        # # If uncomment, do not use GPU.
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU_id
        # print('GPU Available', tf.test.is_gpu_available())
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)

    def train_test(self):
        """Using training/testing set & generated model, do learning & prediction"""

        # Data generation
        self.data = AAAI19Data(self.store_id)
        self.data.run()
        self.nfeat = len(self.data.handcrafted_features)

        print('Number of areas: {}'.format(len(self.data.area_embedding)))

        train_data_size = len(self.data.train_visits)
        test_data_size = len(self.data.test_visits)
        train_censored_data_size = len(self.data.train_censored_visits)
        print(train_data_size, test_data_size, train_censored_data_size)

        self.train_data = self.data.train_data_generator_AAAI19()
        self.test_data = self.data.test_data_generator_AAAI19()
        self.train_censored_data = self.data.train_censored_data_generator_AAAI19()

        # Generate a model
        self.model = AAAI19Model(data=self.data)
        optimizer = tf.keras.optimizers.Adam()

        def calculate_proba(x, interval):
            """ For calculating negative log likelihood losses for censored data."""
            rvbin_label = x[-2]  # revisit binary label
            supp_time = x[-1]  # revisit suppress time  # supp_time = K.cast(K.round(supp_time), dtype='int32')
            kvar_ones = K.ones_like(x[:-2])
            y = subtract([kvar_ones, x[:-2]])  # y = non-revisit rate (1-hazard rate)

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

        def precal(label, input):
            uc_loss_nll_option = 'date'
            self.tmp_tensor = K.concatenate([input, label], axis=-1)
            self.probs_survive = K.map_fn(
                functools.partial(calculate_proba, interval=self.d_interval[uc_loss_nll_option]),
                elems=self.tmp_tensor)

        def uc_c_loss_ce(label):
            """Cross Entropy loss for both cases--censored and uncensored"""
            final_survive_prob = tf.transpose(self.probs_survive)[2]
            final_revisit_prob = tf.subtract(tf.constant(1.0, dtype=tf.float32), final_survive_prob)
            survive_revisit_prob = tf.transpose(tf.stack([final_survive_prob, final_revisit_prob]), name="predict")
            actual_survive_bin = tf.subtract(tf.constant(1.0, dtype=tf.float32), label[:, -2])
            actual_revisit_bin = label[:, -2]
            revisit_binary_categorical = tf.transpose(tf.stack([actual_survive_bin, actual_revisit_bin]))
            result = -tf.reduce_sum(
                revisit_binary_categorical * tf.math.log(tf.clip_by_value(survive_revisit_prob, 1e-10, 1.0)))
            return result

        def uc_loss_nll():
            """Negative log-likelihood loss"""
            prob_revisit_at_z = tf.transpose(self.probs_survive)[0] - tf.transpose(self.probs_survive)[1]
            # If censored -> multiply by 0 -> thus ignored
            prob_revisit_at_z_uncensored = tf.add(tf.multiply(prob_revisit_at_z, tf.transpose(self.probs_survive)[-1]),
                                                  1e-20)
            result = -tf.reduce_sum(K.log(prob_revisit_at_z_uncensored))
            return result

        train_ce_loss = tf.keras.metrics.Mean(name='train_ce_loss')
        train_nll_loss = tf.keras.metrics.Mean(name='train_nll_loss')
        test_ce_loss = tf.keras.metrics.Mean(name='test_ce_loss')
        test_nll_loss = tf.keras.metrics.Mean(name='test_nll_loss')
        train_censored_ce_loss = tf.keras.metrics.Mean(name='train_censored_ce_loss')
        train_censored_nll_loss = tf.keras.metrics.Mean(name='train_censored_nll_loss')

        if FLAGS.all_data:  # Check more often since the whole data is much big
            steps_per_epoch = train_data_size // (10 * FLAGS.batch_size)
        else:
            steps_per_epoch = train_data_size // FLAGS.batch_size

        def train_step(input, label):
            with tf.GradientTape() as tape:
                predictions = self.model(input)
                precal(label, predictions)
                # print(label, input, self.tmp_tensor)  # Easy Debugging by Default Eager Execution!
                ce_loss = uc_c_loss_ce(label)
                nll_loss = uc_loss_nll()
            gradients = tape.gradient(ce_loss+0.2*nll_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            train_ce_loss(ce_loss)
            train_nll_loss(nll_loss)

        def test_step(input, label):
            predictions = self.model(input)
            precal(label, predictions)
            ce_loss = uc_c_loss_ce(label)
            nll_loss = uc_loss_nll()
            test_ce_loss(ce_loss)
            test_nll_loss(nll_loss)

        def train_censored_step(input, label):
            predictions = self.model(input)
            precal(label, predictions)
            ce_loss = uc_c_loss_ce(label)
            nll_loss = uc_loss_nll()
            train_censored_ce_loss(ce_loss)
            train_censored_nll_loss(nll_loss)

        for epoch in range(FLAGS.train_epochs):
            step = 0
            onGoing = True
            while onGoing:
                step+=1
                if step < steps_per_epoch:
                    onGoing = True
                    inputs, labels = next(self.train_data)
                    inputs = tf.cast(inputs, tf.float32)
                    labels = tf.cast(labels, tf.float32)
                    train_step(inputs, labels)
                else:
                    onGoing = False

            # test_inputs, test_labels = next(self.test_data)
            # test_inputs = tf.cast(test_inputs, tf.float32)
            # test_labels = tf.cast(test_labels, tf.float32)
            # test_step(test_inputs, test_labels)
            #
            # train_censored_inputs, train_censored_labels = next(self.train_censored_data)
            # train_censored_inputs = tf.cast(train_censored_inputs, tf.float32)
            # train_censored_labels = tf.cast(train_censored_labels, tf.float32)
            # train_censored_step(train_censored_inputs, train_censored_labels)
            #
            # template = 'Epoch {}, Train-CE-Loss: {:4f}, Train-NLL-Loss: {:4f}, Test-CE-Loss: {:4f}, Test-NLL-Loss: {:4f}, Train-censored-CE-Loss: {:4f}, Train-censored-NLL-Loss: {:4f}'
            # print(template.format(epoch + 1,
            #                       train_ce_loss.result(),
            #                       train_nll_loss.result(),
            #                       test_ce_loss.result(),
            #                       test_nll_loss.result(),
            #                       train_censored_ce_loss.result(),
            #                       train_censored_nll_loss.result(),
            #                       ))

        print('testing begin')
        onGoing = True
        step = 0
        pred_test = []
        while onGoing:
            step += 1
            if step <= test_data_size // FLAGS.batch_size:
                onGoing = True
                test_inputs, test_labels = next(self.test_data)
                test_inputs = tf.cast(test_inputs, tf.float32)
                test_labels = tf.cast(test_labels, tf.float32)
                pred_test.append(self.model(test_inputs))
            else:
                onGoing = False
                remaining_size = test_data_size-(FLAGS.batch_size * (step-1))
                test_inputs, test_labels = next(self.test_data)
                test_inputs = tf.cast(test_inputs, tf.float32)
                test_labels = tf.cast(test_labels, tf.float32)
                pred_test.append(self.model(test_inputs)[:remaining_size, :])
        pred_test = tf.concat(pred_test, axis=0)
        print(pred_test)



        print('train-censored begin')
        onGoing = True
        step = 0
        pred_train_censored = []
        while onGoing:
            step += 1
            if step <= train_censored_data_size // FLAGS.batch_size:
                onGoing = True
                train_censored_inputs, train_censored_labels = next(self.test_data)
                train_censored_inputs = tf.cast(train_censored_inputs, tf.float32)
                train_censored_labels = tf.cast(train_censored_labels, tf.float32)
                pred_train_censored.append(self.model(train_censored_inputs))
            else:
                onGoing = False
                remaining_size = train_censored_data_size - (FLAGS.batch_size * (step - 1))
                train_censored_inputs, train_censored_labels = next(self.test_data)
                train_censored_inputs = tf.cast(train_censored_inputs, tf.float32)
                train_censored_labels = tf.cast(train_censored_labels, tf.float32)
                pred_train_censored.append(self.model(train_censored_inputs)[:remaining_size, :])
        pred_train_censored = tf.concat(pred_train_censored, axis=0)
        print(pred_train_censored)

        #
        # train_censored_inputs, train_censored_labels = next(self.train_censored_data)
        # train_censored_inputs = tf.cast(train_censored_inputs, tf.float32)
        # train_censored_labels = tf.cast(train_censored_labels, tf.float32)
        # pred_train_censored = self.model(train_censored_inputs)
        # # print(K.sum(pred_train_censored, axis=0))

        self.pred_test = pred_test
        self.pred_train_censored = pred_train_censored

        if self.for_baseline == True:
            pass
        else:
            eval = Evaluation()
            eval.evaluate(self.data, pred_test, algo='AAAI19')
            eval.evaluate_train_censored(self.data, pred_train_censored, algo='AAAI19')
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