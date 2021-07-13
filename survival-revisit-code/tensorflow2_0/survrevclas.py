import numpy as np
import tensorflow as tf
from tensorflow import keras
from params import FLAGS
from data import *
from keras.layers.normalization import BatchNormalization
import os
import lifelines
import warnings
warnings.filterwarnings("ignore", category=Warning)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

"""
* Filename: survrevclas.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Our preliminary classification model (NOW DEPRECATED), datas generated from train/test generators are slightly different.
"""


# ToDo: Add diverse metrics to evaluate, implement our first model
class SurvRevClas():
    """Class SurvRevClas describes our preliminary classification model """
    def __init__(self, gpu_id, input_file, log_dir):
        self.gpu_id = gpu_id
        self.input_file = input_file
        self.store_id = input_file[-1]
        self.log_dir = log_dir

        self.train_data = None
        self.test_data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

        self.max_features = 5000
        self.max_len = 30

        self.history = None
        self.result = None

    # def load_data_imdb(self):
    #     (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    #         num_words=self.max_features)
    #     x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.max_len)
    #     x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.max_len)
    #     self.x_train, self.y_train = x_train, y_train
    #     self.x_test, self.y_test = x_test, y_test

    def train_test(self):
        # Some parameters - ToDo: Move those to somewhere
        max_h_features = 5
        gap = 3

        # Data generation
        data = Data(self.store_id)
        data.run()
        print(data.train_visits.head(3))
        print('Number of areas: {}'.format(len(data.area_embedding)))
        max_num_areas = np.max(data.train_visits.areas.apply(len))

        # Generate a Model
        visit_embedding_layer = keras.layers.Embedding(
            input_dim=len(data.visit_embedding),
            output_dim=len(list(data.visit_embedding.values())[0]),
            weights=[np.array(list(data.visit_embedding.values()))],
            input_length=1,
            trainable=False)

        area_embedding_layer = keras.layers.Embedding(
            input_dim=len(data.area_embedding),
            output_dim=len(list(data.area_embedding.values())[0]),
            weights=[np.array(list(data.area_embedding.values()))],
            input_length=max_num_areas,
            trainable=False)

        def simple_attention(target, reference):
            attention = keras.layers.Dense(1, activation=keras.activations.tanh)(reference)
            attention = keras.layers.Reshape((-1,))(attention)
            attention = keras.layers.Activation(keras.activations.softmax)(attention)
            return keras.layers.Dot((1, 1))([target, attention])

        dropout = keras.layers.Dropout(0.2)

        visit_input = keras.Input((1,))
        visit_emb = visit_embedding_layer(visit_input)
        visit_output = keras.layers.Reshape((-1,))(visit_emb)

        areas_input = keras.Input((max_num_areas,))
        areas_emb = area_embedding_layer(areas_input)
        areas_cnn = keras.layers.Conv1D(filters=200, kernel_size=5,
                                        padding='same', activation='relu', strides=1)(areas_emb)
        areas_att = simple_attention(areas_cnn, areas_cnn)
        areas_output = areas_att

        # # without handcrafted features
        # visit_features_input = keras.Input((len(data.handcrafted_features),))
        # concat = keras.layers.Concatenate()([visit_output, areas_output])

        # with handcrafted features
        v_a_emb_concat = keras.layers.Concatenate()([visit_output, areas_output])
        v_a_output = keras.layers.Dense(32, activation=keras.activations.softmax)(v_a_emb_concat)
        visit_features_input = keras.Input((len(data.handcrafted_features),))
        concat = keras.layers.Concatenate()([v_a_output, visit_features_input])
        concat = BatchNormalization()(concat)
        logits = keras.layers.Dense(2, activation=keras.activations.softmax)(concat)
        self.model = keras.Model([visit_input, areas_input, visit_features_input], logits)
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss=keras.losses.categorical_crossentropy,
                           metrics=[keras.metrics.categorical_accuracy])

        print(self.model.summary())

        # Train
        train_data_size = len(data.train_visits)
        self.train_data = data.train_data_generator_clas()
        self.test_data = data.test_data_generator_clas()

        self.history = self.model.fit_generator(
            generator=self.train_data,
            steps_per_epoch=train_data_size//FLAGS.batch_size,
            epochs=5
        )

        self.result = self.model.predict_generator(
            generator=self.test_data,
            steps=1
        )

        print('\nTest prediction result size: {}'.format(len(self.result)))
        print('Train size: {}, Test size: {}'.format(len(data.train_visits), len(data.test_visits)))

        print(data.test_visits[list(data.handcrafted_features)].head(5))

        print('\n------Performance Comparison------')
        print('1) Censored + Uncensored')
        print('  i) Binary classification')
        """Accuracy comparison"""
        golden_binary = data.test_labels['revisit_intention']  # golden_binary: revisit binary labels
        gbvc = golden_binary.value_counts()
        if gbvc[0] / (gbvc[0] + gbvc[1]) > 0.5:
            y_majority_voting = np.zeros(len(self.result))  # If p>0.5, Non-revisit is a majority
        else:
            y_majority_voting = np.ones(len(self.result)) # Opposite case, not happen on our dataset
        y_pred = np.argmax(self.result, axis=1)

        print('    - Accuracy: {:.4} (Majority Voting Baseline)'.format(
            sklearn.metrics.accuracy_score(y_true=golden_binary, y_pred=y_majority_voting)))
        print('    - Accuracy: {:.4} (Our Model)'.format(
            sklearn.metrics.accuracy_score(y_true=golden_binary, y_pred=y_pred)))

        """F-score comparison"""
        print('    - F-score: {:.4} (Majority Voting Baseline)'.format(
            sklearn.metrics.f1_score(y_true=golden_binary, y_pred=y_majority_voting)))
        print('    - F-score: {:.4} (Our Model)'.format(
            sklearn.metrics.f1_score(y_true=golden_binary, y_pred=y_pred)))

        """C-index comparison 
        (Rank difference between actual revisit interval and predicted result)"""
        print('  ii) Rank comparison')
        golden_suppress_time = data.test_suppress_time

        dd = self.result[:, 0]  # =P(censored): large P is equivalent to long revisit_interval)

        # Majority case is non-revisit, therefore P(censored) = 1 for all data samples
        cindex_maj = lifelines.utils.concordance_index(event_times=golden_suppress_time,
                                                       predicted_scores=np.ones(len(self.result)),
                                                       event_observed=(golden_binary == 1))
        cindex = lifelines.utils.concordance_index(event_times=golden_suppress_time,
                                                   predicted_scores=dd,
                                                   event_observed=(golden_binary == 1))
        print('    - C-index: {:.4} (Majority Voting Baseline)'.format(cindex_maj))
        print('    - C-index: {:.4} (Our Model)'.format(cindex))

        print('2) Uncensored Only')
        """MSE comparison 
                (Mean squared error between actual revisit interval and predicted result.
                 This measure is calculated only for revisited case.)"""
        print('  i) Regression Error')
        # ToDo: Change the values after changing objective functions using self.result
        revisited_indices = np.argwhere(~np.isnan(data.train_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.train_labels['revisit_interval'])[revisited_indices]
        pred_revisit_interval = np.array(data.train_labels['revisit_interval'])[revisited_indices]

        mse_maj = 100.000
        mse = sklearn.metrics.mean_squared_error(y_true=golden_revisit_interval,
                                                 y_pred=pred_revisit_interval)

        print('    - MSE: {:.4} (Majority Voting Baseline)'.format(mse_maj))
        print('    - MSE: {:.4} (Our Model)'.format(mse))

        # print('\n(P(censored), Suppress_time, Censored)')
        # for i, j, k in zip(dd[:5], golden_suppress_time[:5], (golden_binary == 1)[:5]):
        #     print('({:4.4f},{:8.3f},{})'.format(i, j, k))

        ## Below code block is for debugging purpose.
        # import code
        # code.interact(local=locals())


    def run(self):
        self.train_test()





