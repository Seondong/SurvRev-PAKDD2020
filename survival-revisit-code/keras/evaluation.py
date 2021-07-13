import numpy as np
import sklearn
import xgboost as xgb
import lifelines
import utils
from params import FLAGS
import csv
import os
from utils import timeit
import random
from MHP import MHP

"""
* Filename: evaluation.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Included methods for performance evaluation.
"""

class Evaluation():
    """Includes some methods for evaluation"""
    def __init__(self):
        self.log_rslt = {}
        self.log_rslt['test'] = {}
        self.log_rslt['train_censored'] = {}
        self.log_rslt['callback_test'] = {}
        self.log_rslt['callback_train_censored'] = {}
        self.run_baselines()

    def run_baselines(self):
        """ Run some baselines"""
        pass

    def calculate_metrics(self, data, y_pred_regr_all, test_case, name):
        """Calculate evaluation metrics - this code block will be used for each evaluation methods"""
        if test_case == 'test':
            data.df_test['tmp'] = list(y_pred_regr_all)
            y_pred_regr_all = data.df_test['tmp']
            del data.df_test['tmp']

            test_uncensored_indices = data.df_test.revisit_interval.notnull()

            y_test_clas = np.asarray(data.df_test['revisit_intention'])
            y_test_regr = np.array(data.df_test['revisit_interval'][test_uncensored_indices])
            y_pred_regr = y_pred_regr_all[test_uncensored_indices]
            y_pred_clas = data.df_test.ts_end + 86400 * y_pred_regr_all > data.last_timestamp

            cindex = lifelines.utils.concordance_index(event_times=data.test_suppress_time,
                                                       predicted_scores=y_pred_regr_all,
                                                       event_observed=(data.test_labels['revisit_intention'] == 1))

        elif test_case == 'train_censored':
            data.df_train_censored['tmp'] = list(y_pred_regr_all)
            y_pred_regr_all = data.df_train_censored['tmp']
            del data.df_train_censored['tmp']

            test_uncensored_indices = data.df_train_censored.revisit_interval.notnull()

            y_test_clas = np.asarray(data.df_train_censored['revisit_intention'])
            y_test_regr = np.array(data.df_train_censored['revisit_interval'][test_uncensored_indices])
            y_pred_regr = y_pred_regr_all[test_uncensored_indices]
            y_pred_clas = data.df_train_censored.ts_end + 86400 * y_pred_regr_all > data.last_timestamp

            cindex = lifelines.utils.concordance_index(event_times=data.train_censored_new_suppress_time,
                                                       predicted_scores=y_pred_regr_all,
                                                       event_observed=(data.train_censored_actual_labels[
                                                                           'revisit_intention'] == 1))

        acc = sklearn.metrics.accuracy_score(y_test_clas, y_pred_clas)
        rmse = utils.root_mean_squared_error(y_test_regr, y_pred_regr)
        fscore = sklearn.metrics.f1_score(y_test_clas, y_pred_clas)

        self.log_rslt[test_case]['acc_{}'.format(name)] = acc
        self.log_rslt[test_case]['fscore_{}'.format(name)] = fscore
        self.log_rslt[test_case]['cindex_{}'.format(name)] = cindex
        self.log_rslt[test_case]['RMSE_{}'.format(name)] = rmse

    def naive_baseline(self, data):
        """Naive baselines for test data"""
        golden_binary = data.test_labels['revisit_intention']
        gbvc = golden_binary.value_counts()
        if gbvc[0] / (gbvc[0] + gbvc[1]) > 0.5:
            y_binary_majority_voting = np.zeros(len(golden_binary))  # If p>0.5, Non-revisit is a majority
        else:
            y_binary_majority_voting = np.ones(len(golden_binary))  # Opposite case, not happen on our dataset
        self.log_rslt['test']['acc_majority_baseline'] = sklearn.metrics.accuracy_score(y_true=golden_binary,
                                                                                        y_pred=y_binary_majority_voting)
        self.log_rslt['test']['fscore_majority_baseline'] = sklearn.metrics.f1_score(y_true=golden_binary,
                                                                                     y_pred=y_binary_majority_voting)
        # Use average suppress time as a prediction result.
        golden_suppress_time = data.test_suppress_time
        self.log_rslt['test']['cindex_majority_baseline'] = lifelines.utils.concordance_index(
            event_times=golden_suppress_time,
            predicted_scores=np.full(golden_suppress_time.shape, np.mean(data.train_suppress_time)),
            event_observed=(golden_binary == 1))

        revisited_indices = np.argwhere(~np.isnan(data.test_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.test_labels['revisit_interval'])[revisited_indices]
        rmse_maj_revisit_interval = np.full(golden_revisit_interval.shape, np.mean(data.train_suppress_time))
        rmse_golden_avg_revisit_interval = np.full(golden_revisit_interval.shape, np.mean(golden_revisit_interval))
        self.log_rslt['test']['RMSE_train_average_baseline'] = utils.root_mean_squared_error(
            y_true=golden_revisit_interval,
            y_pred=rmse_maj_revisit_interval)
        self.log_rslt['test']['RMSE_test_average_baseline'] = utils.root_mean_squared_error(
            y_true=golden_revisit_interval,
            y_pred=rmse_golden_avg_revisit_interval)

    def naive_baseline_for_train_censored(self, data):
        """Naive baselines for train_censored data"""
        golden_binary = data.train_censored_actual_labels[
            'revisit_intention']  # golden_binary: revisit binary labels
        gbvc = golden_binary.value_counts()
        if gbvc[0] / (gbvc[0] + gbvc[1]) > 0.5:
            y_binary_majority_voting = np.zeros(len(golden_binary))  # If p>0.5, Non-revisit is a majority
        else:
            y_binary_majority_voting = np.ones(len(golden_binary))  # Opposite case, not happen on our dataset

        self.log_rslt['train_censored']['acc_majority_baseline'] = sklearn.metrics.accuracy_score(
            y_true=golden_binary,
            y_pred=y_binary_majority_voting)

        self.log_rslt['train_censored']['fscore_majority_baseline'] = sklearn.metrics.f1_score(y_true=golden_binary,
                                                                                               y_pred=y_binary_majority_voting)
        golden_suppress_time = data.train_censored_new_suppress_time
        # Use average suppress time as a prediction result.
        self.log_rslt['train_censored']['cindex_majority_baseline'] = lifelines.utils.concordance_index(
            event_times=golden_suppress_time,
            predicted_scores=np.full(golden_suppress_time.shape, np.mean(data.train_suppress_time)),
            event_observed=(golden_binary == 1))

        revisited_indices = np.argwhere(~np.isnan(data.train_censored_actual_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.train_censored_actual_labels['revisit_interval'])[revisited_indices]
        rmse_maj_revisit_interval = np.full(golden_revisit_interval.shape, np.mean(data.train_suppress_time))
        rmse_golden_avg_revisit_interval = np.full(golden_revisit_interval.shape, np.mean(golden_revisit_interval))
        self.log_rslt['train_censored']['RMSE_train_average_baseline'] = utils.root_mean_squared_error(
            y_true=golden_revisit_interval,
            y_pred=rmse_maj_revisit_interval)
        self.log_rslt['train_censored']['RMSE_test_average_baseline'] = utils.root_mean_squared_error(
            y_true=golden_revisit_interval,
            y_pred=rmse_golden_avg_revisit_interval)

    def poisson_process_baseline(self, data):
        """Poisson process baseline

        Customer's visit can be mapped to bus arrival event to bus-stop,
        Observation time can be mapped to passenger's arrival to bus-stop
        Inter-arrival time (revisit interval): Exponential distribution.
        Need to consider memoryless property of exponential distribution """

        data_start = min(data.df_train.ts_start)
        for test_case in ['test', 'train_censored']:
            # Case for test instances in testing timeframe
            if test_case == 'test':
                # left-censored time for testing set, which can be equivalent to 1/lambda(=mu) for each user
                mu = (data.df_test.ts_start - data_start)
                mu /= data.df_test.nvisits
                y_pred_regr_all = mu.apply(np.random.exponential) / 86400

            # Case for train_censored instances for their prediction
            elif test_case == 'train_censored':
                # For train_censored set, we have an observation until t1 (last time of the train data),
                # so we can make an equation as in the below line.
                mu = max(data.df_train.ts_end) - data_start
                mu /= data.df_train_censored.nvisits
                y_pred_regr_all = mu.apply(np.random.exponential) / 86400
            self.calculate_metrics(data, y_pred_regr_all, test_case=test_case, name='pp_baseline')

    def hawkes_process_baseline(self, data):
        """Hawkes process baseline

        Extended version of Poisson Process with self-simulation and time-decaying
        Used [Hawkes](https://github.com/stmorse/hawkes) libary"""

        d = {}
        for i, j in enumerate(sorted(list(set(data.df_train.wifi_id.astype(int))))):
            d[j] = i
        data.df_train.wifi_id = data.df_train.wifi_id.astype(int).apply(lambda x: d[x])
        train_start_time = min(data.df_train.ts_start)
        train_end_time = max(data.df_train.ts_end)
        test_start_time = min(data.df_test.ts_start)
        test_end_time = max(data.df_test.ts_end)
        num_days = (train_end_time - train_start_time) / 86400

        """Train phase"""
        # For referring train data in the prediction, we first collect train customer's visit rate and self-stimulation rate.
        tdata = np.array([(data.df_train.ts_start - train_start_time) / 86400, data.df_train.wifi_id]).transpose()
        # Split by wifi_id (sorted) -> To put each trajectory into EM model separately
        data_splitted = np.split(tdata, np.where(np.diff(tdata[:, 1]))[0] + 1)

        all_params = []
        for edata in data_splitted:
            edata[:, 1] = 0

            m = np.array([len(edata) / num_days])
            a = np.array([[0.1]])
            w = 0.5

            P = MHP(mu=m, alpha=a, omega=w)
            P.data = edata

            # EM algorithm to estimate parameters
            ahat, mhat = P.EM(a, m, w, verbose=False)

            # Save new parameters to MHP instance
            P.mu = m
            P.alpha = ahat
            #     print('{}, {:2f}, {}, {}'.format(len(edata), num_days, ahat, m))
            all_params.append((ahat, m, P))  # Keep ahat, mhat from training set  (shuld be mhat instead m if Hawkes EM function works with an additional input of time horizon t)

        """Test case"""
        # Predicting revisit of test customers by referring visit rate and self-stimulation rate from train customers.
        y_pred_regr_all_test = []
        for t in list(data.df_test.ts_end):
            remaining_time = (test_end_time - t) / 86400
            ahat, _, _ = random.choice(all_params)
            mhat = np.array([86400 / (t - train_start_time)])

            #     print('Remaining time: {}, Ahat: {}, Mhat: {}'.format(remaining_time, ahat, mhat))
            if np.isnan(ahat):
                ahat = np.array([[0]])
            P = MHP(mu=mhat, alpha=ahat, omega=w)
            P.generate_next_event()
            #     print(P.data)
            try:
                rint = P.data[0][0]
                rbin = int(remaining_time >= rint)
            except IndexError:
                rint = np.inf
                rbin = 0
            y_pred_regr_all_test.append(rint)
        self.calculate_metrics(data, y_pred_regr_all_test, test_case='test', name='hp_baseline')

        """Train censored case"""
        y_pred_regr_all_train_censored = []
        for t, params in zip(list(data.df_train_censored.ts_end), all_params):
            remaining_time = (test_end_time - test_start_time) / 86400
            no_event_time = (test_start_time - t) / 86400

            P = params[-1]
            rel_time = (test_start_time - train_start_time) / 86400
            #     print(t, params, P.data, rel_time)
            #     print('Rate: ', P.get_rate(rel_time, 0))
            #     print('Difference between original rate: {}'.format(P.get_rate(rel_time, 0) - params[1][0]))

            P.mu = params[1]
            # Ad-hoc getaway
            if np.isnan(params[0]):
                P.alpha = np.array([[0]])
            else:
                P.alpha = params[0]
            P.generate_next_event()


            try:
                rint = P.data[0][0]
                rbin = int(remaining_time >= rint)
            except IndexError:
                rint = np.inf
                rbin = 0
            y_pred_regr_all_train_censored.append(rint)

        self.calculate_metrics(data, y_pred_regr_all_train_censored, test_case='train_censored', name='hp_baseline')

    def icdm_baseline(self, data):
        """Several baselines are implemented without considering survival analysis (ICDM'18)"""
        # Initializing values
        acc = 0.0
        rmse = 100000
        cindex = 0.5
        fscore = 0.0

        for task in ['regr', 'clas']:
            if task == 'regr':
                # Dataset for Regression task: Using only uncensored data
                train_uncensored_indices = data.df_train.revisit_interval.notnull()
                test_uncensored_indices = data.df_test.revisit_interval.notnull()

                df_train_regr = data.df_train[train_uncensored_indices]
                df_test_regr = data.df_test[test_uncensored_indices]
                df_test_regr_all = data.df_test

                X_train_regr = np.asarray(df_train_regr[data.handcrafted_features])
                X_test_regr = np.asarray(df_test_regr[data.handcrafted_features])
                X_test_regr_all = np.asarray(df_test_regr_all[data.handcrafted_features])

                y_train_regr = np.asarray(df_train_regr['revisit_interval'])
                y_test_regr = np.asarray(df_test_regr['revisit_interval'])
                y_test_regr_all = np.array(df_test_regr_all['revisit_interval'])

                clf_regr = xgb.XGBRegressor()
                clf_regr = clf_regr.fit(X_train_regr, y_train_regr)

                # Predict and calculate rmse for uncensored data
                y_pred_regr = clf_regr.predict(X_test_regr)
                rmse = utils.root_mean_squared_error(y_test_regr, y_pred_regr)

                # Predict and calculate c-index for all data
                y_pred_regr_all = clf_regr.predict(X_test_regr_all)
                cindex = lifelines.utils.concordance_index(event_times=data.test_suppress_time,
                                                   predicted_scores=y_pred_regr_all,
                                                   event_observed=(data.test_labels['revisit_intention'] == 1))
            if task == 'clas':
                # Dataset for Classification task: All data (censored + uncensored data)

                df_train_clas = data.df_train
                df_test_clas = data.df_test

                X_train_clas = np.asarray(df_train_clas[data.handcrafted_features])
                X_test_clas = np.asarray(df_test_clas[data.handcrafted_features])

                y_train_clas = np.asarray(df_train_clas['revisit_intention'])
                y_test_clas = np.asarray(df_test_clas['revisit_intention'])

                clf_clas = xgb.XGBClassifier()
                clf_clas = clf_clas.fit(X_train_clas, y_train_clas)

                # Prediction
                y_pred_clas = clf_clas.predict(X_test_clas)

                acc = sklearn.metrics.accuracy_score(y_test_clas, y_pred_clas)
                fscore = sklearn.metrics.f1_score(y_test_clas, y_pred_clas)

        self.log_rslt['test']['acc_xgb_baseline'] = acc
        self.log_rslt['test']['fscore_xgb_baseline'] = fscore
        self.log_rslt['test']['cindex_xgb_baseline'] = cindex
        self.log_rslt['test']['RMSE_xgb_baseline'] = rmse

    def icdm_baseline_for_train_censored(self, data, task='regr'):
        """icdm_baseline for train_censored data"""
        # Initializing values
        acc = 0.0
        rmse = 100000
        cindex = 0.5
        fscore = 0.0

        for task in ['regr', 'clas']:
            if task == 'regr':
                # Dataset for Regression task: Using only uncensored data
                train_uncensored_indices = data.df_train.revisit_interval.notnull()
                # Test object: Censored data from train, which finally revisited during the testing period
                test_uncensored_indices = data.df_train_censored.revisit_interval.notnull()

                # For convenience, we use 'test' in the variable name.
                df_train_regr = data.df_train[train_uncensored_indices]
                df_test_regr = data.df_train_censored[test_uncensored_indices]
                df_test_regr_all = data.df_train_censored

                X_train_regr = np.asarray(df_train_regr[data.handcrafted_features])
                X_test_regr = np.asarray(df_test_regr[data.handcrafted_features])
                X_test_regr_all = np.asarray(df_test_regr_all[data.handcrafted_features])

                y_train_regr = np.asarray(df_train_regr['revisit_interval'])
                y_test_regr = np.asarray(df_test_regr['revisit_interval'])
                y_test_regr_all = np.array(df_test_regr_all['revisit_interval'])

                clf_regr = xgb.XGBRegressor()
                clf_regr = clf_regr.fit(X_train_regr, y_train_regr)

                # Predict and calculate rmse for uncensored test data
                y_pred_regr = clf_regr.predict(X_test_regr)
                rmse = utils.root_mean_squared_error(y_test_regr, y_pred_regr)

                # Predict and calculate c-index for all data
                y_pred_regr_all = clf_regr.predict(X_test_regr_all)
                cindex = lifelines.utils.concordance_index(event_times=data.train_censored_new_suppress_time,
                                                   predicted_scores=y_pred_regr_all,
                                                   event_observed=(data.train_censored_actual_labels['revisit_intention'] == 1))
            if task == 'clas':
                # Dataset for Classification task: All data (censored + uncensored data)
                df_train_clas = data.df_train
                df_test_clas = data.df_train_censored

                X_train_clas = np.asarray(df_train_clas[data.handcrafted_features])
                X_test_clas = np.asarray(df_test_clas[data.handcrafted_features])

                y_train_clas = np.asarray(df_train_clas['revisit_intention'])
                y_test_clas = np.asarray(df_test_clas['revisit_intention'])

                clf_clas = xgb.XGBClassifier()
                clf_clas = clf_clas.fit(X_train_clas, y_train_clas)

                # Prediction
                y_pred_clas = clf_clas.predict(X_test_clas)

                acc = sklearn.metrics.accuracy_score(y_test_clas, y_pred_clas)
                fscore = sklearn.metrics.f1_score(y_test_clas, y_pred_clas)

        self.log_rslt['train_censored']['acc_xgb_baseline'] = acc
        self.log_rslt['train_censored']['fscore_xgb_baseline'] = fscore
        self.log_rslt['train_censored']['cindex_xgb_baseline'] = cindex
        self.log_rslt['train_censored']['RMSE_xgb_baseline'] = rmse

    def traditional_survival_analysis_baseline(self, data):
        """Several traditional survival baselines are implemented considering censored data"""
        # Initializing values
        acc = 0.0
        rmse = 100000
        cindex = 0.5
        fscore = 0.0

        # Tracking indices for uncensored data - for RMSE evaluation
        test_uncensored_indices = data.df_test.revisit_interval.notnull()

        # Using all data
        remaining_attributes = list(data.handcrafted_features) + ['suppress_time', 'revisit_intention']
        df_train = data.df_train[remaining_attributes]
        df_test = data.df_test[remaining_attributes]

        # import code
        # code.interact(local=locals())
        # Using Cox Proportional Hazards model
        cph = lifelines.CoxPHFitter()

        # `event_col` refers to whether the 'death' events was observed: 1 if observed, 0 else (censored)
        cph.fit(df_train, 'suppress_time', event_col='revisit_intention', show_progress=False)

        # Prediction & Calculate C-index
        y_pred_regr = cph.predict_expectation(df_test[df_test.columns.drop(['revisit_intention', 'suppress_time'])])
        cindex = lifelines.utils.concordance_index(df_test['suppress_time'], y_pred_regr[0],
                                                   event_observed=df_test['revisit_intention'])

        # Calculate RMSE
        y_pred_regr_uncensored = y_pred_regr[test_uncensored_indices]
        y_test_uncensored = df_test[test_uncensored_indices]['suppress_time']
        rmse = utils.root_mean_squared_error(y_test_uncensored, y_pred_regr_uncensored)

        # Calculate Accuracy & F-score
        y_binary_pred = data.test_visits.ts_end + (86400 * np.array(y_pred_regr).reshape(-1)) <= data.last_timestamp
        y_test_clas = np.asarray(df_test['revisit_intention'])
        acc = sklearn.metrics.accuracy_score(y_test_clas, y_binary_pred)
        fscore = sklearn.metrics.f1_score(y_test_clas, y_binary_pred)

        self.log_rslt['test']['acc_cph_baseline'] = acc
        self.log_rslt['test']['fscore_cph_baseline'] = fscore
        self.log_rslt['test']['cindex_cph_baseline'] = cindex
        self.log_rslt['test']['RMSE_cph_baseline'] = rmse


    def traditional_survival_analysis_baseline_for_train_censored(self, data):
        """Several traditional survival baselines are implemented considering censored data"""
        # Initializing values
        acc = 0.0
        rmse = 100000
        cindex = 0.5
        fscore = 0.0

        # Tracking indices for uncensored data - for RMSE evaluation
        test_uncensored_indices = data.df_train_censored.revisit_interval.notnull()

        # Using all data
        remaining_attributes = list(data.handcrafted_features) + ['suppress_time', 'revisit_intention']
        df_train = data.df_train[remaining_attributes]
        # For convenience, we use 'test' in the variable name.
        df_test = data.df_train_censored[remaining_attributes]

        # Using Cox Proportional Hazards model
        cph = lifelines.CoxPHFitter()
        # `event_col` refers to whether the 'death' events was observed: 1 if observed, 0 else (censored)
        cph.fit(df_train, 'suppress_time', event_col='revisit_intention', show_progress=False)

        # Prediction & Calculate C-index
        y_pred_regr = cph.predict_expectation(df_test[df_test.columns.drop(['revisit_intention', 'suppress_time'])])
        cindex = lifelines.utils.concordance_index(df_test['suppress_time'], y_pred_regr[0],
                                                   event_observed=df_test['revisit_intention'])

        # Calculate RMSE
        y_pred_regr_uncensored = y_pred_regr[test_uncensored_indices]
        y_test_uncensored = df_test[test_uncensored_indices]['suppress_time']
        rmse = utils.root_mean_squared_error(y_test_uncensored, y_pred_regr_uncensored)

        # Calculate Accuracy & F-score
        y_binary_pred = data.train_censored_visits.ts_end + (86400 * np.array(y_pred_regr).reshape(-1)) <= data.last_timestamp
        # To get the binary label for train censored data regards to test timestamp.
        y_test_clas = np.asarray(df_test['revisit_intention'])
        acc = sklearn.metrics.accuracy_score(y_test_clas, y_binary_pred)
        fscore = sklearn.metrics.f1_score(y_test_clas, y_binary_pred)

        self.log_rslt['train_censored']['acc_cph_baseline'] = acc
        self.log_rslt['train_censored']['fscore_cph_baseline'] = fscore
        self.log_rslt['train_censored']['cindex_cph_baseline'] = cindex
        self.log_rslt['train_censored']['RMSE_cph_baseline'] = rmse

    @timeit
    def evaluate(self, data, pred_result):
        """ Evaluation results """
        print('\n----Final evaluation results----')
        print('Train size: {}, Test size: {}'.format(len(data.train_visits), len(data.test_visits)))

        # Prepare useful intermediary results
        pred_revisit_probability, pred_revisit_interval = utils.cal_expected_interval(pred_result)

        print('\n------Performance Comparison (Test: First visitor case)------')
        print('1) Censored + Uncensored')
        print('  i) Binary classification')

        """Accuracy:
        In a regression scheme, binary classification can be done
        by comparing (observation time) and (visit time + predicted interval)"""
        golden_binary = data.test_labels['revisit_intention']  # golden_binary: revisit binary labels
        # True if the predicted revisit time is in the test dataframe, otherwise False.
        y_binary_pred = data.test_visits.ts_end+(86400*pred_revisit_interval.reshape(-1)) <= data.last_timestamp
        self.log_rslt['test']['acc_ours'] = sklearn.metrics.accuracy_score(y_true=golden_binary, y_pred=y_binary_pred)
        self.print_results_console(test_case='test', metric='acc')

        """F-score: """
        self.log_rslt['test']['fscore_ours'] = sklearn.metrics.f1_score(y_true=golden_binary, y_pred=y_binary_pred)
        self.print_results_console(test_case='test', metric='fscore')

        """C-index: 
        Rank difference between actual revisit interval and predicted result"""
        print('  ii) Rank comparison')
        golden_suppress_time = data.test_suppress_time
        censored_probability = 1 - pred_revisit_probability  # =P(censored): large P is equivalent to long revisit_interval)

        self.log_rslt['test']['cindex_ours_interval'] = lifelines.utils.concordance_index(event_times=golden_suppress_time,
                                                   predicted_scores=pred_revisit_interval,
                                                   event_observed=(golden_binary == 1))
        self.log_rslt['test']['cindex_ours_censored_prob'] = lifelines.utils.concordance_index(event_times=golden_suppress_time,
                                                   predicted_scores=censored_probability,
                                                   event_observed=(golden_binary == 1))
        self.print_results_console(test_case='test', metric='cindex')

        print('2) Uncensored Only')
        """RMSE: 
        Root Mean squared error between actual revisit interval and predicted result.
        This measure is calculated only for revisited case.)"""
        print('  i) Regression Error')

        revisited_indices = np.argwhere(~np.isnan(data.test_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.test_labels['revisit_interval'])[revisited_indices]

        self.log_rslt['test']['RMSE_ours'] = utils.root_mean_squared_error(y_true=golden_revisit_interval,
                                                 y_pred=pred_revisit_interval[revisited_indices])
        self.print_results_console(test_case='test', metric='RMSE')

    @timeit
    def evaluate_train_censored(self, data, pred_result):
        """ Evaluation results for censored data"""
        print('\n----Final evaluation results for censored data from training set----')
        print('Train size: {}, Train Censored size: {}'.format(len(data.train_visits), len(data.train_censored_actual_labels)))

        # Prepare useful intermediary results
        pred_revisit_probability, pred_revisit_interval = utils.cal_expected_interval(pred_result)

        print('\n------Performance Comparison (Train censored case)------')
        print('1) Censored + Uncensored')
        print('  i) Binary classification')
        # Accuracy
        golden_binary = data.train_censored_actual_labels['revisit_intention']  # golden_binary: revisit binary labels
        # True if the predicted revisit time is in the test dataframe, otherwise False.
        y_binary_pred = data.train_censored_visits.ts_end + (86400 * pred_revisit_interval.reshape(-1)) <= data.last_timestamp
        self.log_rslt['train_censored']['acc_ours'] = sklearn.metrics.accuracy_score(y_true=golden_binary, y_pred=y_binary_pred)
        self.print_results_console(test_case='train_censored', metric='acc')

        # F-score
        self.log_rslt['train_censored']['fscore_ours'] = sklearn.metrics.f1_score(y_true=golden_binary, y_pred=y_binary_pred)
        self.print_results_console(test_case='train_censored', metric='fscore')

        # C-index
        print('  ii) Rank comparison')
        golden_suppress_time = data.train_censored_new_suppress_time
        censored_probability = 1 - pred_revisit_probability  # =P(censored): large P is equivalent to long revisit_interval)

        self.log_rslt['train_censored']['cindex_ours_interval'] = lifelines.utils.concordance_index(
            event_times=golden_suppress_time,
            predicted_scores=pred_revisit_interval,
            event_observed=(golden_binary == 1))
        self.log_rslt['train_censored']['cindex_ours_censored_prob'] = lifelines.utils.concordance_index(
            event_times=golden_suppress_time,
            predicted_scores=censored_probability,
            event_observed=(golden_binary == 1))
        self.print_results_console(test_case='train_censored', metric='cindex')

        print('2) Uncensored Only')
        # RMSE
        print('  i) Regression Error')
        revisited_indices = np.argwhere(~np.isnan(data.train_censored_actual_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.train_censored_actual_labels['revisit_interval'])[revisited_indices]
        self.log_rslt['train_censored']['RMSE_ours'] = utils.root_mean_squared_error(y_true=golden_revisit_interval,
                                                                           y_pred=pred_revisit_interval[
                                                                               revisited_indices])
        self.print_results_console(test_case='train_censored', metric='RMSE')

    def callback_test_evaluate(self, data, pred_result):
        """Evaluate test set to check whether the model is overfitting or not"""

        # Calculate pred results
        pred_revisit_probability, pred_revisit_interval = utils.cal_expected_interval(pred_result)
        print('    * Test set')

        # Accuracy & F-score (for test)
        golden_binary = data.test_labels['revisit_intention']  # golden_binary: revisit binary labels
        y_binary_pred = data.test_visits.ts_end + (86400 * pred_revisit_interval.reshape(-1)) <= data.last_timestamp
        self.log_rslt['callback_test']['acc_ours'] = sklearn.metrics.accuracy_score(y_true=golden_binary, y_pred=y_binary_pred)
        self.log_rslt['callback_test']['fscore_ours'] = sklearn.metrics.f1_score(y_true=golden_binary, y_pred=y_binary_pred)
        print('      - Accuracy: {:.4} (Our Model)'.format(self.log_rslt['callback_test']['acc_ours']))
        print('      - F-score: {:.4} (Our Model)'.format(self.log_rslt['callback_test']['fscore_ours']))

        # C-index (for test)
        golden_suppress_time = data.test_suppress_time
        self.log_rslt['callback_test']['cindex_ours_interval'] = lifelines.utils.concordance_index(
            event_times=golden_suppress_time,
            predicted_scores=pred_revisit_interval,
            event_observed=(golden_binary == 1))
        print('      - C-index: {:.4} (Our Model)'.format(self.log_rslt['callback_test']['cindex_ours_interval']))

        # RMSE (for test)
        revisited_indices = np.argwhere(~np.isnan(data.test_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.test_labels['revisit_interval'])[revisited_indices]
        self.log_rslt['callback_test']['RMSE_ours'] = utils.root_mean_squared_error(y_true=golden_revisit_interval, y_pred=pred_revisit_interval[revisited_indices])
        print('      - RMSE: {:.4} (Our Model)'.format(self.log_rslt['callback_test']['RMSE_ours']))

    def callback_train_censored_evaluate(self, data, pred_result):
        """Evaluate train_censored set to check whether the model is overfitting or not"""

        # Calculate pred results
        pred_revisit_probability, pred_revisit_interval = utils.cal_expected_interval(pred_result)
        print('    * Censored data from training set')

        # Accuracy & F-score (for train_censored)
        golden_binary = data.train_censored_actual_labels['revisit_intention']
        y_binary_pred = data.train_censored_visits.ts_end + (86400 * pred_revisit_interval.reshape(-1)) <= data.last_timestamp
        self.log_rslt['callback_train_censored']['acc_ours'] = sklearn.metrics.accuracy_score(y_true=golden_binary, y_pred=y_binary_pred)
        self.log_rslt['callback_train_censored']['fscore_ours'] = sklearn.metrics.f1_score(y_true=golden_binary, y_pred=y_binary_pred)
        print('      - Accuracy: {:.4} (Our Model)'.format(self.log_rslt['callback_train_censored']['acc_ours']))
        print('      - F-score: {:.4} (Our Model)'.format(self.log_rslt['callback_train_censored']['fscore_ours']))

        # C-index (for train_censored)
        golden_suppress_time = data.train_censored_new_suppress_time
        self.log_rslt['callback_train_censored']['cindex_ours_interval'] = lifelines.utils.concordance_index(
            event_times=golden_suppress_time,
            predicted_scores=pred_revisit_interval,
            event_observed=(golden_binary == 1))
        print('      - C-index: {:.4} (Our Model)'.format(
            self.log_rslt['callback_train_censored']['cindex_ours_interval']))

        # RMSE (for train_censored)
        revisited_indices = np.argwhere(~np.isnan(data.train_censored_actual_labels['revisit_interval']))
        golden_revisit_interval = np.array(data.train_censored_actual_labels['revisit_interval'])[revisited_indices]
        self.log_rslt['callback_train_censored']['RMSE_ours'] = utils.root_mean_squared_error(y_true=golden_revisit_interval, y_pred=pred_revisit_interval[revisited_indices])
        print('      - RMSE: {:.4} (Our Model)'.format(self.log_rslt['callback_train_censored']['RMSE_ours']))

    def print_output_callback(self, additional_inputs_callback):
        """Print callback results to log files"""

        # Prepare items to print
        list_keys = list(additional_inputs_callback.keys()) + \
                    ['callback_test_' + item for item in list(self.log_rslt['callback_test'].keys())] + \
                    ['callback_train_censored_' + item for item in list(self.log_rslt['callback_train_censored'].keys())]
        list_results = list(additional_inputs_callback.values()) + \
                       list(self.log_rslt['callback_test'].values()) + \
                       list(self.log_rslt['callback_train_censored'].values())
        assert len(list_keys) == len(list_results)

        # Setup a directory for saving experimental results
        if not os.path.exists(FLAGS.exp_result_dir_path):
            os.makedirs(FLAGS.exp_result_dir_path)
        if not os.path.exists(FLAGS.callback_result_dir_path):
            os.makedirs(FLAGS.callback_result_dir_path)

        callback_result_file_path = '{}{}.csv'.format(FLAGS.callback_result_dir_path, additional_inputs_callback['exp_id'])
        exists = os.path.isfile(callback_result_file_path)

        with open(callback_result_file_path, 'a') as ffc:
            wr_callback = csv.writer(ffc, dialect='excel')
            if exists:
                wr_callback.writerow(list_results)
            else:
                wr_callback.writerow(list_keys)
                wr_callback.writerow(list_results)

    def print_results_console(self, test_case, metric):
        """Print results to console"""
        if metric == 'acc':
            full_name = 'Accuracy'
        elif metric == 'fscore':
            full_name = 'F-score'
        elif metric == 'cindex':
            full_name = 'C-index'
        elif metric == 'RMSE':
            full_name = 'RMSE'

        def _if_exist_then_print(metric, key_name, print_name):
            key = '{}_{}'.format(metric, key_name)
            try:
                print('      - {}: {:.4} ({})'.format(full_name, self.log_rslt[test_case][key], print_name))
            except KeyError:
                pass
                # print('      - Result with key {}, does not exist.'.format(key))

        print('    * {}'.format(full_name))
        if metric != 'RMSE':
            _if_exist_then_print(metric, key_name='majority_baseline', print_name='Majority Voting Baseline')
        else:
            _if_exist_then_print(metric, key_name='train_average_baseline', print_name='Train Average Baseline')
            _if_exist_then_print(metric, key_name='test_average_baseline', print_name='Test Average Baseline')
        _if_exist_then_print(metric, key_name='pp_baseline', print_name='PP Baseline')
        _if_exist_then_print(metric, key_name='hp_baseline', print_name='HP Baseline')
        _if_exist_then_print(metric, key_name='xgb_baseline', print_name='ICDM\'18 Baseline - Feature Restricted Ver')
        _if_exist_then_print(metric, key_name='cph_baseline', print_name='Traditional SA - CPH Baseline')
        if metric != 'cindex':
            _if_exist_then_print(metric, key_name='ours', print_name='Our Model')
        else:
            _if_exist_then_print(metric, key_name='ours_interval', print_name='Our Model - calculated by pred_revisit_interval')
            _if_exist_then_print(metric, key_name='ours_censored_prob', print_name='Our Model - calculated by pred_revisit_probability')

    def print_output(self, additional_inputs_all):
        """Print results to log files"""

        # Prepare items to print
        list_keys = list(additional_inputs_all.keys()) + \
                    ['test_' + item for item in list(self.log_rslt['test'].keys())] + \
                    ['train_censored_' + item for item in list(self.log_rslt['train_censored'].keys())]
        list_results = list(additional_inputs_all.values()) + \
                       list(self.log_rslt['test'].values()) + \
                       list(self.log_rslt['train_censored'].values())
        assert len(list_keys) == len(list_results)

        # Setup a directory for saving experimental results
        if not os.path.exists(FLAGS.exp_result_dir_path):
            os.makedirs(FLAGS.exp_result_dir_path)
        if not os.path.exists(FLAGS.all_result_dir_path):
            os.makedirs(FLAGS.all_result_dir_path)

        exists = os.path.isfile(FLAGS.all_result_file_path)

        with open(FLAGS.all_result_file_path, 'a') as ff:
            wr_all = csv.writer(ff, dialect='excel')
            if exists:
                wr_all.writerow(list_results)
            else:
                wr_all.writerow(list_keys)
                wr_all.writerow(list_results)