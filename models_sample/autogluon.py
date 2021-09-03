import logging
import os

try:
    import autogluon
except ImportError:
    import pip
    pip.main(['install', '--user', 'autogluon'])
    import autogluon

from kaggle.api.kaggle_api_extended import KaggleApi

import numpy as np
import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor

logging.info('Load dataset')
train_data = TabularDataset('/kaggle/input/tabular-playground-series-sep-2021/train.csv').drop('id', axis=1)
test_data = TabularDataset('/kaggle/input/tabular-playground-series-sep-2021/test.csv').drop('id', axis=1)

logging.info('Training predictor')
predictor = TabularPredictor(label='claim', eval_metric='roc_auc')
predictor.fit(train_data=train_data, time_limit=14400, presets='best_quality', verbosity=2)

logging.info('Predict Probabilitie')
pred_probs = predictor.predict_proba(test_data)
oof_probs = predictor.predict_proba(train_data.drop(columns=['claim']))

train= pd.read_csv('/kaggle/input/tabular-playground-series-sep-2021/train.csv', sep=',')
train = train.set_index('id')

test= pd.read_csv('/kaggle/input/tabular-playground-series-sep-2021/test.csv', sep=',')
test = test.set_index('id')

pred_probs = pred_probs.set_index(test.index)

sub_sample = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2021/sample_solution.csv', sep=',')
sub_sample = sub_sample.set_index('id')

sub_sample['claim']=pred_probs[1]

sub_sample = sub_sample.reset_index()
sub_sample.to_csv('submission.csv',index=False)

oof_probs.to_csv('oof.csv',index=False)

with open('/root/.kaggle/kaggle.json','w') as fauth:
    fauth.write('{"username":"leonardokrivickas","key":"d2276cf310db144d2b96ebe54568b651"}')
os.system("chmod 600 /root/.kaggle/kaggle.json")

api = KaggleApi()
api.authenticate()
api.competition_submit('submission.csv','API Submission','tabular-playground-series-sep-2021', quiet=True)
