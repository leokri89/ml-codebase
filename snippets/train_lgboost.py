import lightgbm as lgb


def train_lgb_model(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1", "rmse"},
        "learning_rate": 0.01,
        "num_leaves": 23,
        "num_iterations": 10000,
        "verbosity": -1,
    }
    m = lgb.train(
        params,
        train_set=lgb_train,
        valid_sets=lgb_eval,
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    return m
