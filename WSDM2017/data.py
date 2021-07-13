import pandas as pd
import numpy as np
import gensim
import scipy
import sklearn
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from params import FLAGS
import keras
import os
import time
from keras.backend.tensorflow_backend import set_session


class Data():
    def __init__(self, input_file):
        self.input_file = input_file
        self.store_name = 'store_A'
        self.data_path = '{}{}/'.format(FLAGS.release_path, self.store_name)

        self.word2vec_model = None
        self.doc2vec_model = None

        self.area_embedding = None
        self.visit_embedding = None

        self.train_labels = None
        self.test_labels = None
        self.train_visits = None
        self.test_visits = None

        self.df_train = None
        self.df_test = None

        self.train_suppress_time = None
        self.test_suppress_time = None
        self.handcrafted_features = None

        if self.store_name == 'store_A':
            self.last_timestamp = FLAGS.last_timestamp_A
        elif self.store_name == 'store_B':
            self.last_timestamp = FLAGS.last_timestamp_B
        elif self.store_name == 'store_C':
            self.last_timestamp = FLAGS.last_timestamp_C
        elif self.store_name == 'store_D':
            self.last_timestamp = FLAGS.last_timestamp_D
        elif self.store_name == 'store_E':
            self.last_timestamp = FLAGS.last_timestamp_E

    def preprocess_df(self):
        train_labels = pd.read_csv(self.data_path + 'train_labels.tsv', sep='\t')
        test_labels = pd.read_csv(self.data_path + 'test_labels.tsv', sep='\t')
        train_visits = pd.read_csv(self.data_path + 'train_visits.tsv', sep='\t')
        test_visits = pd.read_csv(self.data_path + 'test_visits.tsv', sep='\t')
        wifi_sessions = pd.read_csv(self.data_path + 'wifi_sessions.tsv', sep='\t')

        t3 = time.time()
        wifi_sessions = wifi_sessions.set_index('index')

        def _add_infos(df):
            df['l_index'] = df['indices'].apply(lambda x: [int(y) for y in x.split(';')])

            newidx = [item for sublist in list(df.l_index) for item in sublist]
            tmpdf = wifi_sessions.loc[newidx]
            traj_lens = df.l_index.apply(len)

            tmp_areas = list(tmpdf['area'])
            tmp_dt = list(tmpdf['dwell_time'])
            tmp_ts_start = list(np.array(tmpdf['ts']))
            tmp_ts_end = list(np.array(tmpdf['ts']) + np.array(tmp_dt))  # end time

            rslt_dt = []
            rslt_areas = []
            rslt_ts_start = []
            rslt_ts_end = []

            i = 0
            for x in traj_lens:
                rslt_dt.append(tmp_dt[i:i + x])
                rslt_areas.append(tmp_areas[i:i + x])
                rslt_ts_start.append(min(tmp_ts_start[i:i+x]))
                rslt_ts_end.append(max(tmp_ts_end[i:i+x]))
                i += x

            df['dwell_times'] = rslt_dt
            df['areas'] = rslt_areas
            df['ts_start'] = rslt_ts_start
            df['ts_end'] = rslt_ts_end
            assert all(df.ts_end - df.ts_start >= 0)
            return df

        train_visits = _add_infos(train_visits)
        test_visits = _add_infos(test_visits)
        assert len(test_visits.tail(3)) == 3
        t4 = time.time()
        print('_add_infos running time: {:.4}'.format(t4 - t3))

        model = gensim.models.Word2Vec(list(train_visits['areas']), size=5, min_count=7)
        assert abs(1-scipy.spatial.distance.cosine(model.wv.vectors[0], model.wv.vectors[-1])) <= 1  # cosine similarity

        def _visualize():
            """ Visualize embedded zone(sensor) vectors."""
            viz_words = len(model.wv.vectors)
            word_vector = model.wv.vectors
            tsne = sklearn.manifold.TSNE(n_components=2)
            embed_tsne = tsne.fit_transform(word_vector)
            fig, ax = plt.subplots(figsize=(6, 6))
            for i in range(viz_words):
                plt.scatter(*embed_tsne[i, :], s=2, alpha=0.6, color='b')
                plt.annotate(model.wv.index2word[i], (embed_tsne[i, 0], embed_tsne[i, 1]), alpha=0.6, fontsize=7)
            plt.savefig('tsne_zones_{}.png'.format(self.store_name))

        _visualize()

        """ Applying Doc2Vec to utilize embedded visit ID.
            Reference: https://jusonn.github.io/blog/2018/04/27/Doc2vec-gensim-튜토리얼"""
        def _add_infos2(df):
            """ The purpose of this internal method is to make trajectories to string
            for using Gensim library without further configuration."""
            df['areas_str'] = df['areas'].apply(lambda x: ' '.join(x))
            return df

        train_visits = _add_infos2(train_visits)
        test_visits = _add_infos2(test_visits)

        train_visits['wifi_id'] = train_visits['wifi_id'].astype(str)
        test_visits['wifi_id'] = test_visits['wifi_id'].astype(str)

        def _read_corpus(df, tokens_only=False):
            for i in range(len(df)):
                try:
                    if tokens_only:
                        yield df.areas[i]
                    else:
                        yield gensim.models.doc2vec.TaggedDocument(df.areas[i], [df.wifi_id[i]])
                except:
                    pass

        n_corpus = list(_read_corpus(train_visits))

        model2 = gensim.models.doc2vec.Doc2Vec(n_corpus, dm=1, vector_size=200, min_count=1, epochs=10, hs=0)
        model2.train(n_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        self.word2vec_model = model
        self.doc2vec_model = model2

        # Set Keras
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        area_embedding = {i: j for i, j in zip(model2.wv.index2word, list(model2.wv.vectors))}
        visit_embedding = {i: j for i, j in zip(model2.docvecs.index2entity, list(model2.docvecs.vectors_docs))}

        area_index = {j: i for i, j in enumerate(area_embedding.keys())}
        visit_index = {j: i for i, j in enumerate(visit_embedding.keys())}

        pad_val_area = max(area_index.values())+1
        pad_val_visit = max(visit_index.values())+1

        area_embedding['pad'] = np.zeros(200)
        visit_embedding['pad'] = np.zeros(200)

        num_area_thres = int(np.percentile(train_visits.areas.apply(len), q=99.5))
        train_visits['areas'] = train_visits['areas'].apply(lambda x: x[:num_area_thres] if len(x) > num_area_thres else x)
        train_visits['areas'] = train_visits['areas'].apply(lambda x: x + ['pad']*(num_area_thres-len(x)))
        test_visits['areas'] = test_visits['areas'].apply(lambda x: x[:num_area_thres] if len(x) > num_area_thres else x)
        test_visits['areas'] = test_visits['areas'].apply(lambda x: x + ['pad']*(num_area_thres-len(x)))

        pad_val_area = max(area_index.values())+1
        pad_val_visit = max(visit_index.values()) + 1

        train_visits['area_indices'] = train_visits['areas'].apply(lambda x: [area_index.get(key, pad_val_area) for key in x])
        test_visits['area_indices'] = test_visits['areas'].apply(lambda x: [area_index.get(key, pad_val_area) for key in x])
        train_visits['visit_indices'] = train_visits['wifi_id'].apply(lambda key: visit_index.get(key, pad_val_visit))
        test_visits['visit_indices'] = test_visits['wifi_id'].apply(lambda key: visit_index.get(key, pad_val_visit))

        self.area_embedding = area_embedding
        self.visit_embedding = visit_embedding

        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_visits = train_visits
        self.test_visits = test_visits

    def add_handcrafted_features(self):
        """ Add some important handcrafted features from ICDM'18 (Kim et al) """

        def statistical_feature_generator(x):
            """ Sample code to generate features"""
            fs = []

            total_dwell_time = sum(x['dwell_times'])  # total dwell time
            avg_dwell_time = np.mean(x['dwell_times'])
            num_area_trajectory_have = len(x['dwell_times'])  # the number of area
            num_unique_area_sensed = len(set(x['areas']))  # the number of unique areas

            fs.append(total_dwell_time)
            fs.append(avg_dwell_time)
            fs.append(num_area_trajectory_have)
            fs.append(num_unique_area_sensed)

            return fs

        def add_statistical_features(train_visits):
            df = train_visits.copy()

            features = df.apply(lambda x: statistical_feature_generator(x), axis=1)
            feature_name = ['total_dwell_time', 'avg_dwell_time', 'num_area', 'num_unique_area']

            fdf = pd.DataFrame(list(np.asarray(features)), index=features.index, columns=feature_name)

            # Combine feature values to the dataframe
            df = pd.concat([df, fdf], axis=1)

            # Relative dates
            df['date_rel'] = df['date'] - min(df.date)
            del fdf

            return df

        def add_visit_counts():
            """ Code block to track the previous number of visits.
            (Actually, the number of visits including the current one. """
            self.train_visits['tmp'] = 1
            self.train_visits['nvisits'] = self.train_visits.groupby(['wifi_id'])['tmp'].cumsum()

            wid_nvisit = self.train_visits.iloc[list(self.train_visits['wifi_id'].drop_duplicates(keep='last').index)][
                ['wifi_id', 'nvisits']]
            d_wid_nvisit = {}
            for wid, nvisit in zip(wid_nvisit['wifi_id'], wid_nvisit['nvisits']):
                d_wid_nvisit[wid] = nvisit

            self.test_visits['tmp'] = 1
            self.test_visits['prev_vcount'] = self.test_visits['wifi_id'].apply(lambda x: d_wid_nvisit.get(x, 0))
            self.test_visits['nvisits'] = self.test_visits.groupby(['wifi_id'])['tmp'].cumsum() + self.test_visits['prev_vcount']

            del self.train_visits['tmp'], self.test_visits['tmp'], self.test_visits['prev_vcount'], wid_nvisit

        self.train_visits = add_statistical_features(self.train_visits)
        self.test_visits = add_statistical_features(self.test_visits)
        add_visit_counts()
        self.train_visits['wifi_id']

    def add_unk_prev_interval(self):
        """ Generate 'unk_prev_interval' as an additional feature for each visit.
         The unknown previous interval can be a feature to represent the previous customer revisit interval.

            It comes from the idea that the first-time visitors are possibly repeated visitors, and using the gap
         is important to predict his/her next revisit time.

            For example, if the customer visits the store the first time during the test phase, then we can assume that
         this customer has a low chance to revisit, since the customer didn't visit during the training period.

            This unk_prev_interval feature can be a more accurate feature than the date_rel, and can be a
         combination between date_rel and the previous revisit interval.

            The value of unk_prev_interval is equivalent to min(previous interval, visit_time - data collection start time).

            We didn't use revisit_interval labels, since test labels exist for evaluation purpose.

        Output
        -------
        train/test_visits['unk_prev_interval'] : data series
        """

        # For train_visits:
        first_ts_start = min(self.train_visits['ts_start'])  # if no prev visits, this time can be a standard.
        c1 = self.train_visits.groupby(['wifi_id'])['ts_start']
        c2 = self.train_visits.groupby(['wifi_id'])['ts_end']
        # Record previous interval(n~n+1) for n+1th visit. If a previous visit is unavailable, the value is nan.
        train_prev_revisit_interval = (c1.shift(periods=0) - c2.shift(periods=1))
        # Left observation time using a standard time.
        train_left_observation_time = self.train_visits['ts_start'] - np.full(len(self.train_visits), first_ts_start)
        # Result: For unavailable case, impute observation time. For available case, use previous interval.
        self.train_visits['unk_prev_interval'] = np.minimum(train_prev_revisit_interval.fillna(1e10), train_left_observation_time)

        # For test visits:
        # To calculate unk_revisit_interval for test visits, save the last timestamp of visits appeared on train data.
        self.train_visits['tmp'] = 1
        self.train_visits['nvisits'] = self.train_visits.groupby(['wifi_id'])['tmp'].cumsum()
        # Keep the last timestamp for each wifi_id
        wid_ts_end = self.train_visits.iloc[list(self.train_visits['wifi_id'].drop_duplicates(keep='last').index)][
            ['wifi_id', 'ts_end']]
        d_wid_ts_end = {} # In a dictionary format
        for wid, ts_end in zip(wid_ts_end['wifi_id'], wid_ts_end['ts_end']):
            d_wid_ts_end[wid] = ts_end
        # Save the previous timestamp for each test instance, save the observation time if not available.
        test_left_appeared_time = self.test_visits['wifi_id'].apply(lambda x: d_wid_ts_end.get(x, first_ts_start))
        # Result
        self.test_visits['unk_prev_interval'] = self.test_visits['ts_start'] - test_left_appeared_time

        # Remove intermediary output
        del self.train_visits['tmp'], self.train_visits['nvisits']

        # Check validity
        assert all(self.train_visits['ts_start']-np.full(len(self.train_visits), first_ts_start)
                   >= self.train_visits['unk_prev_interval'])
        assert all(self.test_visits['ts_start'] - np.full(len(self.test_visits), first_ts_start)
                   >= self.test_visits['unk_prev_interval'])

        # Change units from seconds to days
        self.train_visits['unk_prev_interval'] /= 86400
        self.test_visits['unk_prev_interval'] /= 86400

    def add_suppress_time(self):
        """ Generate 'suppress_time' column for evaluation.
         Suppress time is equivalent to the min(event time, observation time) and it is used in calculating c-index.
         Integrated files: df_train, df_test can be used for other experiments in the future."""

        def _add_suppress_time(df):
            last_ts_end = max(df['ts_end'])
            df['tmp_suppress_time'] = [(last_ts_end - x) / 86400 for x in df['ts_end']]
            df['suppress_time'] = np.maximum(df['revisit_interval'].fillna(0),
                                             df['revisit_interval'].isnull() * df['tmp_suppress_time'])
            del df['tmp_suppress_time']
            return df

        self.df_train = pd.concat([self.train_visits, self.train_labels[['revisit_intention', 'revisit_interval']]],
                                  axis=1)
        self.df_test = pd.concat([self.test_visits, self.test_labels[['revisit_intention', 'revisit_interval']]],
                                 axis=1)
        self.df_train = _add_suppress_time(self.df_train)
        self.df_test = _add_suppress_time(self.df_test)
        self.train_suppress_time = self.df_train['suppress_time']
        self.test_suppress_time = self.df_test['suppress_time']

    def remove_unnecessary_features(self):
        """ To control the input features easily"""
        def _remove_unnecessary_features(df):
            unnecessary_attributes = ['date_rel']  # could be more
            all_attributes = list(df.columns)
            for attribute in unnecessary_attributes:
                try:
                    all_attributes.remove(attribute)
                except:
                    pass
            df = df[all_attributes]
            return df

        self.train_visits = _remove_unnecessary_features(self.train_visits)
        self.test_visits = _remove_unnecessary_features(self.test_visits)
        self.df_train = _remove_unnecessary_features(self.df_train)
        self.df_test = _remove_unnecessary_features(self.df_test)

        handcrafted_features_starting_idx = list(self.train_visits.columns).index('visit_indices') + 1
        self.handcrafted_features = self.train_visits.columns[handcrafted_features_starting_idx:]

    def normalize(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.train_visits[self.handcrafted_features] = scaler.fit_transform(self.train_visits[self.handcrafted_features])
        self.test_visits[self.handcrafted_features] = scaler.fit_transform(self.test_visits[self.handcrafted_features])
        self.df_train[self.handcrafted_features] = scaler.fit_transform(self.df_train[self.handcrafted_features])
        self.df_test[self.handcrafted_features] = scaler.fit_transform(self.df_test[self.handcrafted_features])

    def train_data_generator_clas(self):
        def __gen__():
            col_names = self.train_visits.columns
            while True:
                idxs = list(self.train_visits.index)
                np.random.shuffle(idxs)
                for idx in idxs:
                    visit = self.train_visits.iloc[idx]
                    label = self.train_labels.iloc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], label['revisit_intention']

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
            yield [batch[0].reshape(-1, 1), batch[1], batch[2]], keras.utils.to_categorical(batch[-1], 2)

    def test_data_generator_clas(self):
        def __gen__():
            while True:
                idxs = list(self.test_visits.index)
                for idx in idxs:
                    visit = self.test_visits.iloc[idx]
                    label = self.test_labels.iloc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], label['revisit_intention']

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(len(self.test_visits))))]
            yield [batch[0].reshape(-1, 1), batch[1], batch[2]]

    def train_data_generator_regr(self):
        def __gen__():
            while True:
                idxs = list(self.df_train.index)
                np.random.shuffle(idxs)
                for idx in idxs:
                    visit = self.train_visits.iloc[idx]
                    label = self.df_train.iloc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], label['suppress_time']

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
            yield [batch[0].reshape(-1, 1), batch[1], batch[2]], batch[-1].reshape(-1, 1)

    def test_data_generator_regr(self):
        def __gen__():
            while True:
                idxs = list(self.df_test.index)
                for idx in idxs:
                    visit = self.test_visits.iloc[idx]
                    label = self.df_test.iloc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], label['suppress_time']

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(len(self.test_visits))))]
            yield [batch[0].reshape(-1, 1), batch[1], batch[2]]

    def train_data_generator(self):
        def __gen__():
            while True:
                idxs = list(self.df_train.index)
                np.random.shuffle(idxs)
                for idx in idxs:
                    visit = self.train_visits.iloc[idx]
                    label = self.df_train.iloc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], \
                          [label[ft] for ft in ['revisit_intention', 'suppress_time']]
                              # label['revisit_intention'], label['suppress_time']

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(FLAGS.batch_size)))]
            yield [batch[0].reshape(-1, 1), batch[1], batch[2]], batch[-1]

            # batch[-1], batch[-2] 맞게 나옴
            # batch[-2], batch[-1] loss가 10만이상, MAE는 0.47 이렇게 나옴
            # batch[-2], batch[-1].reshape(-1, 1)

    def test_data_generator(self):
        def __gen__():
            while True:
                idxs = list(self.df_test.index)
                for idx in idxs:
                    visit = self.test_visits.iloc[idx]
                    label = self.df_test.iloc[idx]
                    yield visit['visit_indices'], visit['area_indices'], \
                          [visit[ft] for ft in self.handcrafted_features], \
                          [label[ft] for ft in ['revisit_intention', 'suppress_time']]

        gen = __gen__()

        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(len(self.test_visits))))]
            yield [batch[0].reshape(-1, 1), batch[1], batch[2]]

    def run(self):
        self.preprocess_df()
        self.add_handcrafted_features()
        self.add_unk_prev_interval()
        self.add_suppress_time()
        self.remove_unnecessary_features()
        # self.normalize()     # Not necessary after using BatchNormalization on concat layer (survrevclas.py)
        print('All columns: {}'.format(list(self.train_visits.columns)))

