import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow2_0.params import FLAGS
from data import *
from keras.layers.normalization import BatchNormalization
import os
import lifelines
import warnings
import scipy
warnings.filterwarnings("ignore", category=Warning)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

"""
* Filename: survrevregr.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Our preliminary regression model (NOW DEPRECATED), datas generated from train/test generators are slightly different.
"""


# ToDo: Add diverse metrics to evaluate, implement our first model
class SurvRevRegr():
    """Class SurvRevRegr describes our preliminary regression model """
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
        v_a_output = keras.layers.Dense(32, activation=keras.activations.tanh)(v_a_emb_concat)
        visit_features_input = keras.Input((len(data.handcrafted_features),))
        concat = keras.layers.Concatenate()([v_a_output, visit_features_input])
        concat = BatchNormalization()(concat)
        logits = keras.layers.Dense(1, activation=keras.activations.exponential)(concat)

        # def customLoss():
        #     return

        self.model = keras.Model([visit_input, areas_input, visit_features_input], logits)
        self.model.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss=keras.losses.mean_squared_error,
                           metrics=['mae'],
                           )

        print(self.model.summary())

        # Train
        train_data_size = len(data.train_visits)
        self.train_data = data.train_data_generator_regr()
        self.test_data = data.test_data_generator_regr()

        self.history = self.model.fit_generator(
            generator=self.train_data,
            steps_per_epoch=train_data_size//FLAGS.batch_size,
            epochs=10
        )

        self.result = self.model.predict_generator(
            generator=self.test_data,
            steps=1
        )

        # import code
        # code.interact(local=locals())

        """ Evaluation results """
        print('\nTrain size: {}, Test size: {}'.format(len(data.train_visits), len(data.test_visits)))
        gbvc = data.test_labels['revisit_intention'].value_counts()
        revisited_indices = np.argwhere(~np.isnan(data.test_labels['revisit_interval'])).reshape(-1)
        golden_revisit_interval = np.array(data.test_labels['revisit_interval'])[revisited_indices]
        print('\nBefore checking the evaluation metrics...')
        print('Prediction result statistics (All): {}'.format(scipy.stats.describe(self.result[:, 0])))
        print('Prediction result statistics ( 0 ): {}'.format(scipy.stats.describe(self.result[:, 0][revisited_indices])))
        print('Prediction result statistics ( 1 ): {}'.format(scipy.stats.describe(self.result[:, 0][-revisited_indices])))
        print('Actual revisit intention stat: 0: {}, 1: {}, Ratio: {:.4}'.format(gbvc[0], gbvc[1], gbvc[1]/(gbvc[0]+gbvc[1])))
        print('Actual revisit interval stat ( 1 ): {}'.format(
            scipy.stats.describe(golden_revisit_interval.reshape(-1))))

        print('\n------Performance Comparison------')
        print('1) Censored + Uncensored')
        print('  i) Binary classification')
        """Accuracy comparison"""
        """In a regression scheme, binary classification can be done 
        by comparing (observation time) and (visit time + predicted interval)"""
        golden_binary = data.test_labels['revisit_intention']  # golden_binary: revisit binary labels
        gbvc = golden_binary.value_counts()
        if gbvc[0] / (gbvc[0] + gbvc[1]) > 0.5:
            y_majority_voting = np.zeros(len(self.result))  # If p>0.5, Non-revisit is a majority
        else:
            y_majority_voting = np.ones(len(self.result)) # Opposite case, not happen on our dataset

        # True if the predicted revisit time is in the test dataframe, otherwise False.
        y_pred = data.test_visits.ts_end+(86400*self.result.reshape(-1)) <= data.last_timestamp

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

        # Use average suppress time as a prediction result.
        cindex_maj = lifelines.utils.concordance_index(event_times=golden_suppress_time,
                                                       predicted_scores=np.full(golden_suppress_time.shape,np.mean(data.train_suppress_time)),
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
        revisited_indices = np.argwhere(~np.isnan(data.test_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.test_labels['revisit_interval'])[revisited_indices]

        pred_revisit_interval = self.result[revisited_indices].reshape(-1, 1)

        mse_maj = sklearn.metrics.mean_squared_error(y_true=golden_revisit_interval,
                                                 y_pred=np.full(golden_revisit_interval.shape,np.mean(data.train_suppress_time)))
        mse = sklearn.metrics.mean_squared_error(y_true=golden_revisit_interval,
                                                 y_pred=pred_revisit_interval)

        print('    - MSE: {:.4} (Majority Voting Baseline)'.format(mse_maj))
        print('    - MSE: {:.4} (Our Model)'.format(mse))

        # print('\nTest prediction result size: {}'.format(len(self.result)))
        # print(data.test_visits[list(data.handcrafted_features)].head(5))

        # print('\n(P(censored), Suppress_time, Censored)')
        # for i, j, k in zip(dd[:5], golden_suppress_time[:5], (golden_binary == 1)[:5]):
        #     print('({:4.4f},{:8.3f},{})'.format(i, j, k))

        # Below code block is for debugging purpose.
        # import code
        # code.interact(local=locals())

        """ QnA
            1) Mean-squared-error를 objective loss로 하였을 때의 예측값은 굉장히 소극적이다, 작은 평균에 작은 stdev 
                   = 재방문을 하는 형태로 예측 -> 마지막 activation layer를 exponential을 줌으로써 해결하였다.
            2) 현재 모델의 경우 0일 때와 1일 때의 경향성이 비슷한데, 이는 modeling objective으로 suppress time만 이용했기 때문이다.
               0,1 binary 정보와 suppress time을 모두 이용해서 loss를 꾸몄다면, 서로 다른 경향성을 나타내도록 학습이 되었을 것이다.
        """


    def run(self):
        self.train_test()





