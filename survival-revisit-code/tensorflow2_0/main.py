import sys
import os
# from survrevclas import *
# from survrevregr import *
from survrevtensorflow2 import *
from params import FLAGS
import multiprocessing

"""
* Filename: main.py
* Implemented by Sundong Kim (sundong.kim@kaist.ac.kr)

Included methods for performance evaluation.
"""

def ss(ids):
    sid, gpuid = ids
    return sid+'-'+gpuid

def multirun(ids):
    sid, gpuid = ids
    survrevk = SurvRev(store_id=sid, GPU_id=gpuid)
    survrevk.run()

def main():
    """Main command of our survival-revisit method.

        Note: This implementation is for ZOYI survival dataset.

        Parameters (example - to fill for other methods like this)
        ----------
        y_true : array, shape = [n_samples] or [n_samples, n_classes]
            True binary labels or binary label indicators.

        y_score : array, shape = [n_samples] or [n_samples, n_classes]
            Target scores, can either be probability estimates of the positive
            class, co

        Returns (ex)
        -------
        auc : float

        Examples (ex)
        --------
        > y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        > roc_auc_score(y_true, y_scores)
        0.75
    """

    print('FLAGS.multiprocessing: {}'.format(FLAGS.multiprocessing))
    print('FLAGS.all_data: {}'.format(FLAGS.all_data))
    print('FLAGS.previous_visits: {}'.format(FLAGS.previous_visits))
    print('FLAGS.train_epochs: {}'.format(FLAGS.train_epochs))


    if FLAGS.multiprocessing:
        p = multiprocessing.Pool(5)
        store_ids = ['store_A', 'store_B', 'store_C', 'store_D', 'store_E']
        GPU_ids = ["5", "1", "2", "3", "0"]

        print(p.map(ss, zip(store_ids, GPU_ids))) # check multiprocessing is working
        p.map(multirun, zip(store_ids, GPU_ids))

    else:
        survrevk = SurvRev(store_id=FLAGS.store_id, GPU_id="0")
        survrevk.run()

        # wsdm = WSDM(store_id=FLAGS.store_id, GPU_id="3")
        # wsdm.run()

        # aaai19 = AAAI19(store_id=FLAGS.store_id, GPU_id="3")
        # aaai19.run()

if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    main()