import logging
import time
import warnings

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
SAMPLE_PATH = "./dataset/sample_submission.csv"
SEED = 10


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://127.0.0.1:5000")


def train_eval_model(
    model, train_features_set, train_target_set, test_features_set, test_target_set
):
    """[summary]

    Args:
        model ([type]): [description]
        train_features_set ([type]): [description]
        train_target_set ([type]): [description]
        test_features_set ([type]): [description]
        test_target_set ([type]): [description]

    Returns:
        [type]: [description]
    """
    model.fit(train_features_set, train_target_set)
    # predicts = model.predict(test_features_set)
    proba_predicts = model.predict_proba(test_features_set)[:, 1]
    auc = roc_auc_score(test_target_set, proba_predicts)
    return model, auc


train = pd.read_csv(TRAIN_PATH, low_memory=False)
target = train["target"].copy()
train.drop(["id", "target"], inplace=True, axis=1)

continuous = list(train.columns[train.dtypes == "float64"])
discrets = list(train.columns[(train.dtypes == "int64")])

skf = StratifiedKFold(n_splits=3, random_state=SEED, shuffle=True)
splits = [[x, y] for x, y in skf.split(train, target)]

mlflow.sklearn.autolog(silent=True, disable=True)

kbd = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="kmeans")
kbd.fit(train[continuous], target)
modified_set = pd.DataFrame(kbd.transform(train[continuous]), columns=continuous)
modified_set[discrets] = train[discrets]

feature_set = modified_set
target_set = target

with mlflow.start_run(nested=True) as run:
    mlflow.sklearn.autolog(silent=True, disable=False)

    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)

    for idx, split in enumerate(splits):
        start_time = time.time()
        idx_train, idx_test = split[0], split[1]

        with mlflow.start_run(run_name=f"split_{str(idx)}", nested=True) as split_run:
            trained_model, auc_result = train_eval_model(
                rfc,
                feature_set.loc[idx_train],
                target_set.loc[idx_train],
                feature_set.loc[idx_test],
                target_set.loc[idx_test],
            )
            mlflow.log_metric("test_roc_auc_score", auc_result)
            mlflow.log_metric("execution_time", time.time() - start_time)

    mlflow.sklearn.autolog(silent=True, disable=True)
