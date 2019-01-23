import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix
from imblearn.over_sampling import SMOTE




if __name__ == '__main__':
    # df1 = pd.read_csv('../../../data/sample_df', index_col=0)
    # df2 = pd.read_csv('../../../data/trial_df_1.csv', index_col=0)
    df3 = pd.read_csv('../../../data/trial_df_2.csv', index_col=0)

    # Create 6 different categories - one for each AWO bucket
    df3['Binary_Bucket'] = df3['AWO_Bucket'].map({0: 0, 1: 0, 2:1, 3:1, 4:1, 5:1})
    #  create quantiles buckets
    quantiles = pd.qcut(df3['AWO%'],4).unique().sort_values()
    mapdict = {value: index for index, value in enumerate(quantiles)}
    df3['quartiles'] = df3['AWO%'].map(mapdict)

    # create different y's to predict on
    y1 = df3.pop('AWO_Bucket')
    y2 = df3.pop('Binary_Bucket')
    y3 = df3.pop('quartiles')
    # get rid of AWO%, can use for regression
    y4 = df3.pop('AWO%')
    X = df3.values


    # XG boost for Binary Bucket classification
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.25, random_state=7)

    # plot unbalanced data
    # pd.value_counts(y_train).plot.bar()
    # plt.title('AWO Class Histogram')
    # plt.xlabel('Class')
    # plt.xticks([0,1], ["AWO < $1,000", "AWO >= $1,000"], rotation='horizontal')
    # plt.ylabel('Frequency')
    # y4.value_counts()
    # plt.savefig('../../images/unbalanced_AWO_bargraph')
    # # plt.show()
    # plt.close()

    print(f"Before OverSampling, counts of label '1': {sum(y_train==1)}")
    print(f"Before OverSampling, counts of label '0': {sum(y_train==0)} \n")

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

    print(f'After OverSampling, the shape of train_X: {X_train_res.shape}')
    print(f'After OverSampling, the shape of train_y: {y_train_res.shape} \n')

    print(f"After OverSampling, counts of label '1': {sum(y_train_res==1)}")
    print(f"After OverSampling, counts of label '0': {sum(y_train_res==0)}")

    # plot balanced data
    # pd.value_counts(y_train_res).plot.bar()
    # plt.title('AWO Class Histogram - After SMOTE')
    # plt.xlabel('Class')
    # plt.xticks([0,1], ["AWO < $1,000", "AWO >= $1,000"], rotation='horizontal')
    # plt.ylabel('Frequency')
    # y4.value_counts()
    # plt.savefig('../../images/balanced_AWO_bargraph')
    # # plt.show()
    # plt.close()


    # fit model to training data
    model = XGBClassifier()
    model.fit(X_train_res, y_train_res)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {round(accuracy*100.0, 2)}")

    # # create confusion matrix
    # conf_mat = confusion_matrix(y_test, predictions)
    # fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
    #                             show_absolute=False,
    #                             show_normed=True,
    #                             colorbar=False)
    # plt.savefig('../../images/ryan_imgs/conf_mat')


    '''
    # XGboost regression
    X_train, X_test, y_train, y_test = train_test_split(X, y4, test_size=0.25, random_state=7)
    # fit model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    r_squared = r2_score(y_test, predictions)
    print(f'r2 score: {r_squared}')
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f'rmse: {rmse}')
    '''
