import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import cluster, decomposition, ensemble, manifold, random_projection, preprocessing
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
