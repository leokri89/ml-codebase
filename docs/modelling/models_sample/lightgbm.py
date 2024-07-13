import pandas as pd
import numpy as np
import random
import time
import os
import gc
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

N_SPLITS = 5
N_ESTIMATORS = 20000
EARLY_STOPPING_ROUNDS = 200
VERBOSE = 1000
SEED = 2021

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv(INPUT + "sample_solution.csv")

features = [col for col in test.columns if 'f' in col]
TARGET = 'claim'

target = train[TARGET].copy()

train['n_missing'] = train[features].isna().sum(axis=1)
test['n_missing'] = test[features].isna().sum(axis=1)

train['std'] = train[features].std(axis=1)
test['std'] = test[features].std(axis=1)

features += ['n_missing', 'std']
n_missing = train['n_missing'].copy()

train[features] = train[features].fillna(train[features].mean())
test[features] = test[features].fillna(test[features].mean())

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

lgb_params = {
    'objective': 'binary',
    'n_estimators': N_ESTIMATORS,
    'random_state': SEED,
    'learning_rate': 5e-3,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.4,
    'reg_alpha': 10.0,
    'reg_lambda': 1e-1,
    'min_child_weight': 256,
    'min_child_samples': 20,
    'importance_type': 'gain',
}

lgb_oof = np.zeros(train.shape[0])
lgb_pred = np.zeros(test.shape[0])
lgb_importances = pd.DataFrame()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X=train, y=n_missing)):
    print(f"===== fold {fold} =====")
    X_train = train[features].iloc[trn_idx]
    y_train = target.iloc[trn_idx]
    X_valid = train[features].iloc[val_idx]
    y_valid = target.iloc[val_idx]
    X_test = test[features]
    
    start = time.time()
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=VERBOSE,
    )
    
    fi_tmp = pd.DataFrame()
    fi_tmp['feature'] = model.feature_name_
    fi_tmp['importance'] = model.feature_importances_
    fi_tmp['fold'] = fold
    fi_tmp['seed'] = SEED
    lgb_importances = lgb_importances.append(fi_tmp)

    lgb_oof[val_idx] = model.predict_proba(X_valid)[:, -1]
    lgb_pred += model.predict_proba(X_test)[:, -1] / N_SPLITS

    elapsed = time.time() - start
    auc = roc_auc_score(y_valid, lgb_oof[val_idx])
    print(f"fold {fold} - lgb auc: {auc:.6f}, elapsed time: {elapsed:.2f}sec\n")

print(f"oof lgb roc = {roc_auc_score(target, lgb_oof)}")

np.save("lgb_oof.npy", lgb_oof)
np.save("lgb_pred.npy", lgb_pred)

submission[TARGET] = lgb_pred
submission.to_csv("submission.csv", index=False)

with open('model_LGBMClassifier.pkl','wb') as file:
    file.write(pickle.dumps(model))
