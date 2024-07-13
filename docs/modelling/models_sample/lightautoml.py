import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import os

try:
    import lightautoml
except ImportError:
    import pip
    pip.main(['install', '--user', 'lightautoml'])
    import lightautoml

logging.info('Importing Libs')
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from kaggle.api.kaggle_api_extended import KaggleApi
import torch

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco


logging.info('Setting Variables')
N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 3*3600
TARGET_NAME = 'claim'

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

logging.info('Load Datasets')
train_data = pd.read_csv('../input/tabular-playground-series-sep-2021/train.csv')
test_data = pd.read_csv('../input/tabular-playground-series-sep-2021/test.csv')

task = Task('binary', )
roles = {'target': TARGET_NAME, 'drop': ['id']}

logging.info('Executing Search')
automl = TabularAutoML(task = task, 
                       timeout = TIMEOUT,
                       cpu_limit = N_THREADS,
                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
                       general_params = {'use_algos': [['linear_l2', 'lgb', 'lgb_tuned']]}
                      )

logging.info('Fitting predict')
oof_pred = automl.fit_predict(train_data, roles=roles)

test_pred = automl.predict(test_data)

logging.info('Creating submition file')
samp_sub = pd.DataFrame(test_data['id'])
samp_sub[TARGET_NAME] = test_pred.data[:, 0]
samp_sub.to_csv('submission.csv', index = False)

logging.info('Uploading submition')
with open('/root/.kaggle/kaggle.json','w') as fauth:
    fauth.write('{"username":"leonardokrivickas","key":"d2276cf310db144d2b96ebe54568b651"}')
os.system("chmod 600 /root/.kaggle/kaggle.json")

api = KaggleApi()
api.authenticate()
api.competition_submit('submission.csv','API Submission','tabular-playground-series-sep-2021', quiet=True)
