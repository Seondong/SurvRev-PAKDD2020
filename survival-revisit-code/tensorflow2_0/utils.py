import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Activation, Dot
import numpy as np
import sklearn
from math import sqrt
import time

"""
* Filename: utils.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Included several code snippets used in our project.
"""

def simple_attention(target, reference):
    """ Simple Attention Layer """
    attention = Dense(1, activation='tanh')(reference)
    attention = Reshape((-1,))(attention)
    attention = Activation('softmax')(attention)
    result = Dot((1, 1))([target, attention])
    return result


def cal_expected_interval_single(single_visitor_result):
    """
    Calculate expected interval \sigma xf(x)
    Input: Result of one user - shape: (365,)
    (Each element denotes hazard rate - Revisit rate in each timestamp conditioned that the customer is not revisited yet.)
    """
    surv_rate = np.insert(1 - single_visitor_result, 0, 1) #
    surv_prob = np.cumprod(surv_rate, dtype=float)
    revisit_prob = surv_prob[:-1] * single_visitor_result
    rint = np.array(range(365)) + 0.5
    pred_revisit_probability = np.sum(revisit_prob)  # Revisit probability for single user within 365 days
    pred_revisit_interval = np.sum(rint * revisit_prob) + (1-pred_revisit_probability)*365   # Predicted revisit interval for single user
    return pred_revisit_probability, revisit_interval


def cal_expected_interval(result):
    """
    Calculate expected interval \sigma xf(x)
    Expanded version of cal_expected_interval_single
    Input: Result of all users - shape: (test_size, 365)
    """
    surv_rate = 1 - np.hstack((np.zeros(shape=(len(result), 1)), result))  #
    surv_prob = np.cumprod(surv_rate, axis=1, dtype=float)
    revisit_prob = surv_prob[:, :-1] * result
    rint = np.stack([np.array(range(365)) + 0.5 for _ in range(len(result))], axis=0)
    pred_revisit_probability = np.sum(revisit_prob, axis=1)  # Revisit probability for all users within 365 days
    pred_revisit_interval = np.sum(rint * revisit_prob, axis=1) +(1-pred_revisit_probability)*365  # Predicted revisit interval for all users
    print('PROB: ', np.round(pred_revisit_probability[:20], 4))
    print('INT: ', np.round(pred_revisit_interval[:20], 4))
    print('MEAN: {:.2}, {:.2}'.format(np.mean(pred_revisit_probability), np.mean(pred_revisit_interval)))
    print('STD: {:.2}, {:.2}'.format(np.std(pred_revisit_probability), np.std(pred_revisit_interval)))
    return pred_revisit_probability, pred_revisit_interval


def root_mean_squared_error(y_true, y_pred):
    """Root Mean Squared Error"""
    return sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))


def timeit(method):
    """Measure running time.
       ToDo: Later save on log file (kwarg)
       Source: https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def get_stats(df):
    vc = df.revisit_intention.value_counts()
    d = {}
    d['censored_ratio'] = vc[0] / (vc[0]+vc[1])
    d['revisit_ratio'] = vc[1] / (vc[0]+vc[1])
    return d
