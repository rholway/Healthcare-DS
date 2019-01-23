from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import Lars, LassoLars, LassoLarsCV

from sklearn.metrics import roc_auc_score, r2_score, mean_squared_log_error, mean_squared_error

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
import operator


def select_model(X,y,estimators,score_type='neg_mean_squared_error',cv=10):
    estimator_score={}
    for model in estimators:
        # print(f'Cross validating {model.__class__.__name__}')
        estimator_score[model.__class__.__name__]=(-1*np.mean((cross_validate(model,X,y,scoring=score_type,cv=cv))['test_score']))**.5
    return estimator_score

def tune_model(X,y,model, cv=10):
    estimator_score={}
    if model.__class__.__name__ == 'Lasso':
        lass = LassoCV(n_alphas=50,cv=cv).fit(X,y)
        best_alpha = lass.alpha_
    elif model.__class__.__name__ == 'Ridge':
        alphas = np.logspace(-3,3,50)
        ridge = RidgeCV(alphas=alphas, cv=cv).fit(X,y)
        best_alpha = ridge.alpha_
    elif model.__class__.__name__ == 'ElasticNet':
        enet = ElasticNetCV(n_alphas=50, cv=cv).fit(X,y)
        best_alpha = enet.alpha_
    elif model.__class__.__name__  == 'LassoLars':
        ll = LassoLarsCV(fit_intercept=True, max_n_alphas=50).fit(X,y)
        best_alpha = ll.alpha_
    return best_alpha


if __name__ == '__main__':
    # df = pd.read_excel('../../../data/DemoData.xlsx')
    # df = pd.read_csv('../../../data/sample_df')
    df = pd.read_csv('../../../data/trial_df_1.csv')
    df['AWO%'] = df['Transaction_Amount'] / df['NPSR']
    df.drop(['Unnamed: 0', 'Transaction_Amount', 'AWO_Bucket', 'NPSR',
        'Last_Payment_Amount', 'Length_Of_Stay'], axis=1, inplace=True)

    df.dropna(inplace=True)
    y = df.pop('AWO%').values
    X = df.values



    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print('Tuning Ridge')
    ridge_alpha = tune_model(X_train,y_train, Ridge())
    print(f'Ridge alpha: {ridge_alpha}')

    print('Tuning Lasso')
    lasso_alpha = tune_model(X_train,y_train, Lasso())
    print(f'Lasso alpha: {lasso_alpha}')

    print('Tuning ElasticNet')
    enet_alpha = tune_model(X_train,y_train, ElasticNet())
    print(f'ElasticNet alpha: {enet_alpha}')

    print('Tuning LassoLars')
    ll_alpha = tune_model(X_train,y_train, LassoLars())
    print(f'LassoLars alpha: {ll_alpha}')


    models=[LinearRegression(),Ridge(alpha = ridge_alpha),
            Lasso(alpha = lasso_alpha),ElasticNet(alpha=enet_alpha),
            LassoLars(alpha = ll_alpha)]
    test_est=select_model(X_train,y_train,models)
    print(test_est)
    b = max(test_est, key=test_est.get)
    print(f'Best model is {b} with an RMSE of {round(test_est[b],3)}')
