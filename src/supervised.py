import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import cluster, decomposition, ensemble, manifold, random_projection, preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score, mean_squared_error
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import MultinomialNB
import statsmodels.stats.outliers_influence as oi

class MLClassifier():
    '''
    Instatiate a sklearn classifier object

    methods:

    '''

    def __init__(self, X_arr=None, y_arr=None, arr_path=None):
        '''
        Instantiate object with X and y numpy arrays

        args:

        X_arr (numpy array): full x matrix to use for training and classification
        y_arr (numpy array): full y array to use for model evaluation and training
        arr_path (tuple of length == 2): This tuple must only contain two elements which are both strings \
                                         the first element is the path to the X matrix and the second \
                                         element is the path to the y matrix

        '''
        if arr_path:
            self.X = np.load(arr_path[0])
            self.y = np.load(arr_path[1])
        else:
            self.X = X_arr
            self.y = y_arr

        self.classifier_model = None
        self.grid_search = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def integrate_pca(self, PCAModel, integration_type=None):
        if integration_type == 'pca_only':
            self.X == PCAModel.X_pca
        else:
            self.X = np.hstack((self.X, PCAModel.X_pca))

    def split_data(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)


    def fit(self, classifier, **kwargs):
        '''
        fit data to specified sklearn model

        args:
        classifier (object): sklearn model object
        **kwargs (keyword arguments): key word arguments for specific sklearn model
        '''

        self.classifier_model = classifier(**kwargs)
        self.classifier_model.fit(self.X_train, self.y_train)

    def pred_score(self):
        '''
        Get accuracy of prediction on test data
        '''
        accuracy = self.classifier_model.score(self.X_test, self.y_test)
        print('Accuracy: {}'.format(accuracy))
        return accuracy


    def grid_search(self, classifier, params, set_classifier=False):
        ''' gets a rough idea where the best parameters lie

        args:

        classifier (sklearn object): selected classifier sklearn object to search over
        params (dictionary): dictionary of hyperparameters. keys = parameter name, values = paramater values.
        set_classifier (boolean): if True this method will set the best estimator of the grid search as the classifier that can then be evaluated
        '''

        self.grid_search = GridSearchCV(classifier, params)
        print("Starting grid search")
        self.grid_search.fit(self.X_train, self.y_train)
        grid_params = self.grid_search.best_params_
        grid_score = self.grid_search.best_score_
        print("Coarse search best parameters:")
        for param, val in grid_params.items():
            print("{0:<20s} | {1}".format(param, val))
        print("Coarse search best score: {0:0.3f}".format(grid_score))
        if set_classifier:
            self.classifier = self.grid_search.best_estimator_


    def plot_roc_curve(self, img_type='original'):
        '''
        Plot ROC courve for trained classifier model

        args:
        img_type (string): specify if the image type is padded or original.
        '''

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        probs = self.classifier.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probs[:,1])
        auc_score = round(roc_auc_score(self.y_test, probs[:,1]), 4)
        ax.plot(fpr, tpr, label= f'{self.classifier.__class__.__name__} = {auc_score} AUC')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance')
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_title("ROC plot of Mammogram mass classification with {} images".format(img_type), fontsize=18)
        ax.legend()
        plt.show()
        accuracy = self.classifier.score(self.X_test, self.y_test)
        print('Accuracy: {}'.format(accuracy))

    def save_ml_model(self, path):
        '''
        Save trained classifier model

        args:
        path (string): path including name of model to save
        '''
        with open('{}'.format(path), 'wb') as f:
            Pickle.dump(self.classifier, f)

class MLRegressor():

    def __init__(self, X_arr=None, y_arr=None, arr_path=None):
        '''
        Instantiate object with X and y numpy arrays

        args:

        X_arr (numpy array): full x matrix to use for training and classification
        y_arr (numpy array): full y array to use for model evaluation and training
        arr_path (tuple of length == 2): This tuple must only contain two elements which are both strings \
                                         the first element is the path to the X matrix and the second \
                                         element is the path to the y matrix

        '''
        if arr_path:
            self.X = np.load(arr_path[0])
            self.y = np.load(arr_path[1])
        else:
            self.X = X_arr
            self.y = y_arr

        self.regressor_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def integrate_pca(self, PCAModel, integration_type=None):
        if integration_type == 'pca_only':
            self.X == PCAModel.X_pca
        else:
            self.X = np.hstack((self.X, PCAModel.X_pca))

    def split_data(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

    def fit(self, regressor, **kwargs):
        '''
        fit data to specified sklearn model

        args:
        Regressor (object): sklearn model object
        **kwargs (keyword arguments): key word arguments for specific sklearn model
        '''

        self.regressor_model = regressor(**kwargs)
        self.regressor_model.fit(self.X_train, self.y_train)

    def pred_score(self, other_score=None):
        '''
        Get accuracy of prediction on test data
        '''
        if other_score:
            y_pred = self.regressor_model.predict(self.X_test)
            score = other_score(self.y_test, y_pred)
            print('Score: {}'.format(score))
        else:
            score = self.regressor_model.score(self.X_test, self.y_test)
            print('r^2: {}'.format(score))
            return score

def bucket_numeric_vals(df, column, num_partitions):
    bucketed = pd.qcut(df_full[column], num_partitions)
    bucket_vals = bucketed.unique().sort_values()
    bucket_dictionary = dict(zip(bucket_vals, np.arange(num_partitions).tolist()))
    y_bucket = df_full[column].map(bucket_dictionary)
    return y_bucket

if __name__=='__main__':
    df_full = pd.read_csv('../../navigant_data/final_df_cl_edit.csv')
    df_full.drop('Unnamed: 0', axis=1, inplace=True)
    df_full['target_percentage'] = (df_full['awo_amount'] / df_full['npsr']) *100
    targets = ['locationid', 'awo_bucket', 'region', 'npsr', 'awo_amount', 'target_percentage']
    drop_idx = df_full[df_full['npsr'] != 0].index
    df_full = df_full.iloc[drop_idx]

    df_X = df_full.drop(targets, axis=1)
    y_vals_regression = df_full['target_percentage']


    binary_bucket_y = bucket_numeric_vals(df_full, 'target_percentage', 2)
    quartile_bucket_y = bucket_numeric_vals(df_full, 'target_percentage', 4)
    nci_bucket_y = bucket_numeric_vals(df_full, 'target_percentage', 6)



    # vif(x_vals)
    # df_vif = drop_vif_cols(x_vals, 10)
    # df_new = calculate_vif_(x_vals, 10)

    # regressor = MLRegressor(X_arr=df_X.values, y_arr=y_vals.values)
    # regressor.split_data()
    # regressor.fit(RandomForestRegressor, n_estimators=100)
    # score1 = regressor.pred_score(other_score=mean_squared_error)
    # score2 = regressor.pred_score()

    # gb_regressor = MLRegressor(X_arr=df_X.values, y_arr=y_vals.values)
    # gb_regressor.split_data()
    # gb_regressor.fit(GradientBoostingRegressor)
    # gb_score1 = gb_regressor.pred_score(other_score=mean_squared_error)
    # gb_score2 = gb_regressor.pred_score()

    # fig, ax = plt.subplots(111)
    # ax.plot(regressor.y_test)
    # ax.plot(regressor.regressor_model.predict(y_test))


    # classifier = MLClassifier(X_arr=x_vals, y_arr=y_bucket)
    # classifier.split_data()
    # classifier.fit(RandomForestClassifier, n_estimators=1000)
    # score = classifier.pred_score()
    #
    # feature_imp = np.argsort(classifier.classifier_model.feature_importances_)
    # top_five = list(x_vals.columns[feature_imp[-1:-6:-1]])
