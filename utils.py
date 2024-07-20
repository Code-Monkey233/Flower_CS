from typing import List, Tuple, Dict
import numpy as np
from statsmodels.tsa.arima import ARIMA
from sklearn.linear_model import LogisticRegression
from flwr.common import NDArrays, Metrics, Scalar

def get_model_parameters(model):
    if model.fit_intercept:
        params = [model.coef_, model.intercept_, ]
    else:
        params = [model.coef_, ]
    return params

def set_model_params(model: LogisticRegression, params:NDArrays) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model:LogisticRegression, n_classes:int, n_features:int):
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercepts:
        model.intercept_ = np.zeros((n_classes,))


