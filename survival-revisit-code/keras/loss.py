import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import functools
from params import FLAGS

"""
* Filename: loss.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

* Custom loss functions
* Summary
    - Type of uncensored loss
        - Negative Log Likelihood (Similar to L_z in AAAI'19 DRSA)
            - uc_loss_nll_date:
            - uc_loss_nll_week:
            - uc_loss_nll_month:
            - uc_loss_nll_season:
            - uc_loss_nll_day: Not implemented yet
        - MSE
            - uc_loss_mse: MSE loss
        - Cross Entropy of CDF (Similar to L_uncensored in AAAI'19 DRSA)
    - Type of censored loss
        - Cross Entropy of CDF (Similar to L_uncensored in AAAI'19 DRSA)
        - Ranking difference (Similar to L_2 in AAAI'18 DeepHit)
------------------------
* Output: Total loss (Weighted sum, implemented in survrev.py)
------------------------
* Other Infos:
    - y_pred를 각 bin에 속할 평균이라고 생각하자(365개 timebin)
"""

class CustomLoss():
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.tmp_tensor = None
        self.pred_revisit_probability = None
        self.pred_revisit_interval = None
        self.losses = {}
        self.d_interval = {'date': {'left': -0.5, 'right': 0.5},
                           'week': {'left': -3.5, 'right': 3.5},
                           'month': {'left': -15, 'right': 15},
                           'season': {'left': -45, 'right': 45}}
        self.c_nll_date = None
        self.c_nll_week = None
        self.c_nll_month = None
        self.c_nll_season = None
        self.c_rmse = None
        self.c_ce = None
        self.c_rank = None

    def initialize_data(self, y_true, y_pred):
        """ For initializing intermediate results for calculating losses later.
            - Output: pred_revisit_interval, pred_revisit_probability etc. """
        self.y_true = y_true
        self.y_pred = y_pred
        self.tmp_tensor = K.concatenate([self.y_pred, self.y_true], axis=-1)

    @staticmethod
    def calculate_proba(x, interval):
        """ For calculating negative log likelihood losses for censored data."""
        rvbin_label = x[-2]  # revisit binary label
        supp_time = x[-1]  # revisit suppress time  # supp_time = K.cast(K.round(supp_time), dtype='int32')
        kvar_ones = K.ones_like(x[:-2])
        y = keras.layers.Subtract()([kvar_ones, x[:-2]])  # y = non-revisit rate (1-hazard rate)

        # print('calculate_proba - x shape: ', K.int_shape(x))
        # print('calculate_proba - supp_time shape: ', K.int_shape(supp_time))

        left_bin = K.maximum(supp_time + interval['left'], K.ones_like(
            supp_time))  # reason: y[0:x] cannot be calculated when x < 1, therefore set x as 1 so that y[0:1] = 1
        right_bin = K.minimum(supp_time + interval['right'], K.ones_like(
            supp_time) * 365)  # reason: y[0:x] cannot be calculated when x > 365

        left_bin = K.cast(K.round(left_bin), dtype='int32')
        right_bin = K.cast(K.round(right_bin), dtype='int32')
        supp_time_int = K.cast(K.round(supp_time), dtype='int32')

        p_survive_until_linterval = K.prod(
            y[0:left_bin])  # The instance has to survive for every time step until t
        p_survive_until_rinterval = K.prod(y[0:right_bin])
        p_survive_until_supp_time = K.prod(y[0:supp_time_int])

        result = K.stack(
            [p_survive_until_linterval, p_survive_until_rinterval, p_survive_until_supp_time, rvbin_label])
        return result

    def example_loss(self):
        """Toy example to check this custom loss method is working
        - MSE Loss Example when we do not need to check censorship"""
        loss_name = 'example_loss'
        self.losses[loss_name] = K.mean(K.square(K.mean(self.y_pred, axis=1) - self.y_true[:, -1]), axis=-1)
        return self.losses[loss_name]

    def uc_loss_nll(self, uc_loss_nll_option='date'):
        """Wrapper function for all negative log-likelihood loss"""
        loss_name = 'uc_loss_nll_{}'.format(uc_loss_nll_option)
        probs_survive = K.map_fn(functools.partial(self.calculate_proba, interval=self.d_interval[uc_loss_nll_option]),
                                 elems=self.tmp_tensor, name='survive_rates')
        prob_revisit_at_z = tf.transpose(probs_survive)[0] - tf.transpose(probs_survive)[1]
        # If censored -> multiply by 0 -> thus ignored
        prob_revisit_at_z_uncensored = tf.add(
            tf.multiply(prob_revisit_at_z, tf.transpose(probs_survive)[-1]), 1e-20)
        # num_revisitors_in_batch = K.sum(probs_survive[-1])
        self.losses[loss_name] = -tf.reduce_sum(K.log(prob_revisit_at_z_uncensored))   # / num_revisitors_in_batch
        self.losses[loss_name] = tf.add(self.losses[loss_name], 0, name=loss_name)
        return self.losses[loss_name]

    def uc_loss_nll_date(self):
        return self.uc_loss_nll(uc_loss_nll_option='date')

    def uc_loss_nll_week(self):
        return self.uc_loss_nll(uc_loss_nll_option='week')

    def uc_loss_nll_month(self):
        return self.uc_loss_nll(uc_loss_nll_option='month')

    def uc_loss_nll_season(self):
        return self.uc_loss_nll(uc_loss_nll_option='season')

    def uc_loss_nll_day(self):
        return self.uc_loss_nll(uc_loss_nll_option='day')

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

    def uc_loss_rmse(self):
        """Calculate RMSE loss"""
        loss_name = 'uc_loss_rmse'
        pred_revisit_interval = self.cal_expected_interval_for_loss(self.y_pred)
        squared_error_all = K.square(keras.layers.Subtract()([pred_revisit_interval, self.y_true[:, -1]]))
        squared_error_uc = tf.multiply(squared_error_all, self.y_true[:, -2])  # y_true[-2] is a binary revisit label.
        num_revisitors_in_batch = K.sum(self.y_true[:, -2])
        self.losses[loss_name] = K.sqrt(K.sum(squared_error_uc) / num_revisitors_in_batch)
        self.losses[loss_name] = tf.add(self.losses[loss_name], 0, name=loss_name)
        return self.losses[loss_name]


    # def uc_loss_rmse(self):
    #     """Calculate New version of RMSE loss (For censored case: suppress time += 100) <= Kinda implemented Prof's advice about modifying RMSE """
    #     loss_name = 'uc_loss_rmse'
    #     pred_revisit_interval = self.cal_expected_interval_for_loss(self.y_pred)
    #
    #     # suppress time += 100 for censored case
    #     one_vector = K.ones_like(self.y_true[:, -1], dtype='float32')
    #     censored_identifier = keras.layers.Subtract()([one_vector, self.y_true[:, -2]])
    #     reified_suppress_time = censored_identifier * 100 + self.y_true[:, -1]
    #
    #     squared_error_all = K.square(keras.layers.Subtract()([pred_revisit_interval, reified_suppress_time]))
    #
    #     self.losses[loss_name] = K.sqrt(K.sum(squared_error_all) / FLAGS.batch_size)
    #     self.losses[loss_name] = tf.add(self.losses[loss_name], 0, name=loss_name)
    #     return self.losses[loss_name]

    def uc_c_loss_ce(self):
        """ cross entropy loss (both of uncensored and censored data) ToDo: To Keras Code """
        loss_name = 'uc_c_loss_ce'
        probs_survive = K.map_fn(functools.partial(self.calculate_proba, interval=self.d_interval['date']),
                                 elems=self.tmp_tensor, name='survive_rates')
        final_survive_prob = tf.transpose(probs_survive)[2]
        final_revisit_prob = tf.subtract(tf.constant(1.0, dtype=tf.float32), final_survive_prob)
        survive_revisit_prob = tf.transpose(tf.stack([final_survive_prob, final_revisit_prob]), name="predict")

        # ToDo: Mystery - Why revisit_binary_categorical = keras.utils.to_categorical(self.y_true[:, -2]) does not work?
        actual_survive_bin = tf.subtract(tf.constant(1.0, dtype=tf.float32), self.y_true[:, -2])
        actual_revisit_bin = self.y_true[:, -2]
        revisit_binary_categorical = tf.transpose(tf.stack([actual_survive_bin, actual_revisit_bin]))

        self.losses[loss_name] = -tf.reduce_sum(
            revisit_binary_categorical * tf.math.log(tf.clip_by_value(survive_revisit_prob, 1e-10, 1.0)))
        return self.losses[loss_name]

    def uc_c_loss_rank(self):
        """ Rank loss (both of uncensored and censored data) - Motivated by AAAI'18 """
        loss_name = 'uc_c_loss_rank'
        pred_revisit_interval = K.reshape(self.cal_expected_interval_for_loss(self.y_pred), shape=(-1, 1))

        true_suppress_time = K.reshape(self.y_true[:, -1], shape=(-1, 1))
        true_revisit_bin = K.reshape(self.y_true[:, -2], shape=(-1, 1))

        one_vector = K.ones_like(pred_revisit_interval, dtype='float32')

        # \hat{R}_{ij} = \hat{y}_{u_j} - \hat{y}_{u_i}
        pdiff = K.dot(one_vector, K.transpose(pred_revisit_interval)) - K.dot(pred_revisit_interval, K.transpose(one_vector))
        # R_{ij} = y_{u_j} - y_{u_i}
        tdiff = K.dot(one_vector, K.transpose(true_suppress_time)) - K.dot(true_suppress_time, K.transpose(one_vector))

        # Since we define loss, we should minimize it, so we measure wrong pair -> Hence, tdiff is defined as an opposite way
        pdiff = K.relu(keras.layers.Lambda(lambda x: -x)(K.sign(pdiff)))  # \hat{R}_{ij} = 1 if \hat{y}_{u_j} - \hat{y}_{u_i} < 0
        tdiff = K.relu(keras.layers.Lambda(lambda x: x)(K.sign(tdiff)))
        tdiff = tf.multiply(tdiff, K.dot(true_revisit_bin, K.transpose(one_vector)))  # Substitute censored part to 0

        result_matrix = tf.compat.v1.matrix_band_part(tf.multiply(pdiff, tdiff), 0, -1)
        self.losses[loss_name] = K.sum(result_matrix)
        return self.losses[loss_name]

    def combined_loss(self):
        """Our final loss value"""
        coeff_sum = FLAGS.c_nll_date + FLAGS.c_nll_week + FLAGS.c_nll_month + FLAGS.c_nll_season + FLAGS.c_rmse + FLAGS.c_ce + FLAGS.c_rank

        self.c_nll_date = FLAGS.c_nll_date / coeff_sum
        self.c_nll_week = FLAGS.c_nll_week / coeff_sum
        self.c_nll_month = FLAGS.c_nll_month / coeff_sum
        self.c_nll_season = FLAGS.c_nll_season / coeff_sum
        self.c_rmse = FLAGS.c_rmse / coeff_sum
        self.c_ce = FLAGS.c_ce / coeff_sum
        self.c_rank = FLAGS.c_rank / coeff_sum

        combined_loss = self.c_nll_date * self.uc_loss_nll_date() + \
                        self.c_nll_week * self.uc_loss_nll_week() + \
                        self.c_nll_month * self.uc_loss_nll_month() + \
                        self.c_nll_season * self.uc_loss_nll_season() + \
                        self.c_rmse * self.uc_loss_rmse() + \
                        self.c_ce * self.uc_c_loss_ce() + \
                        self.c_rank * self.uc_c_loss_rank()
        return combined_loss