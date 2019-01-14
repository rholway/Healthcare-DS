import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import cluster, decomposition, ensemble, manifold, random_projection, preprocessing
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import statsmodels.stats.outliers_influence as oi


class PCAModel():

    '''
    Create a PCA model
    Make scree plot
    Make 2d component plot
    '''

    def __init__(self, X_array):
        '''
        Instantiate class with X vals only

        args
        X_array (numpy array): X matrix to apply PCA
        '''
        self.X = X_array
        self.y = None
        self.pca_model = None
        self.X_pca = None

    def make_pca_model(self, n_components=2):
        '''
        Create sklearn pca object

        args:
        n_components (int): integer to specify the number of principal components\
                            to include in the model
                            NOTE: You do not have to specify 2 components to plot your model,\
                            this is accounted for in the plotting method
        '''
        scaler = preprocessing.StandardScaler() #always scale values for PCA
        X_scaled = scaler.fit_transform(self.X)
        self.pca_model = decomposition.PCA(n_components=n_components)
        self.X_pca = self.pca_model.fit_transform(X_scaled)
        return self.X_pca, self.pca_model


    def scree_plot(self, ax, n_components_to_plot=10, title=None):
         """Make a scree plot showing the variance explained (i.e. variance
         of the projections) for the principal components in a fit sklearn
         PCA object.

         args:

         ax (matplot lib object): matplot lib axes object
         n_components_to_plot (int): number of principal components to plot to show\
                                     variance explained for each component
         title (optional) (string): title of plot
         """
         num_components = self.pca_model.n_components_
         ind = np.arange(num_components)
         vals = self.pca_model.explained_variance_ratio_
         ax.plot(ind, vals, color='blue')
         ax.scatter(ind, vals, color='blue', s=50)

         for i in range(num_components):
             ax.annotate(r"{:2.2f}%".format(vals[i]),
                    (ind[i]+0.2, vals[i]+0.005),
                    va="bottom",
                    ha="center",
                    fontsize=12)

         ax.set_xticklabels(ind, fontsize=12)
         ax.set_ylim(0, max(vals) + 0.05)
         ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
         ax.set_xlabel("Principal Component", fontsize=12)
         ax.set_ylabel("Variance Explained (%)", fontsize=12)
         if title:
             ax.set_title(title, fontsize=16)

    def plot_2d(self, y, ax, title=None):
         """Plot an embedding of your dataset onto a 2d plane.

         args:
         y (numpy array): numpy array of tragets corresponding to X_array
         ax (matplotlib axes object): axes to use for plotting 2d PCA plot
         title (optional) (string): title of plot
         """

         X_vals = self.X_pca[:, :2] #grabbing first two principal components to plot in 2d space
         self.y = y
         arr = self.y
         arr[arr == 0] = 2

         x_min, x_max = np.min(X_vals, 0), np.max(X_vals, 0)
         X_final = (X_vals - x_min) / (x_max - x_min)
         ax.axis('off')
         ax.patch.set_visible(False)
         for i in range(X_final.shape[0]):
             plt.text(X_final[i, 0], X_final[i, 1],
                      str(arr[i]),
                      color=plt.cm.Set1(arr[i] / 10),
                      fontdict={'weight': 'bold', 'size': 12})

         ax.set_xticks([]),
         ax.set_yticks([])
         ax.set_ylim([-0.1,1.1])
         ax.set_xlim([-0.1,1.1])
         ax.set_legend()

         if title:
             ax.set_title(title, fontsize=16)

    def visualize(self, path=None):
        '''
        Show or save scree and/or 2d plot depending on what method(s) have been called

        args:
        path (string): path including figure name to save figure to
        '''
        if path:
            plt.savefig(path)
        else:
            plt.show()

class KMeansModel():

    def __init__(self, X):
        self.X = X
        self.kmeans_model = None

    def fit(self, n_clusters=2):
        self.kmeans_model = KMeans(n_clusters=n_clusters)
        self.kmeans_model.fit(self.X)
        return self.kmeans_model

    def elbow_plot(self, max_clust):
        scores = [-(self.fit(n_clusters=n).score(self.X)) for n in range(1, max_clust + 1)]
        plt.plot(range(1, max_clust + 1), scores)
        plt.xlabel('K')
        plt.ylabel('RSS')
        plt.title('RSS given K clusters')
        plt.show()
        plt.close()

    def silhouette_plot(self, max_clust):
        sil_scores = [silhouette_score(self.X, self.fit(n_clusters=n).labels_) for n in range(2, max_clust + 1)]
        plt.plot(range(2, max_clust + 1), sil_scores)
        plt.xlabel('K')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouetter Score vs K')
        plt.show()
        plt.close()


def vif(x_mat):
    for idx, col in enumerate(x_mat.columns):
        print(f"{col}: {oi.variance_inflation_factor(x_mat.values,idx)}")

def drop_vif_cols(x_mat, threshold):
    target_cols = [col for idx, col in enumerate(x_mat.columns) if oi.variance_inflation_factor(x_mat.values,idx) < threshold]
    new = x_mat.loc[:, target_cols]
    return new

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [oi.variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def count_classes(df, target_col, type='downsample'):
    class_counts = df.groupby(target_col).count().iloc[:, 0].values.tolist()
    classes = df.groupby(target_col).count().iloc[:, 0].index.values
    if type == 'downsample':
        min_idx = np.argmin(class_counts)
        min_val = class_counts[min_idx]
        min_class = classes[min_idx]
        target_ls = classes[classes != min_class].tolist()
        final_df = df[df[target_col] == min_class]
        for element in target_ls:
            # df[df[target_col] == class].sample(n=min_val)
            final_df = pd.concat([final_df, df[df[target_col] == element].sample(n=min_val)])
    # else:
    #     max_idx = np.argmax(class_counts)
    #     max_val = class_counts[max_idx]
    #     max_class = classes[max_idx]
    #     target_ls = classes[classes != max_class].tolist()
    #     final_df = df[df[target_col] == max_class]
    #     for element in target_ls:
    #         # df[df[target_col] == class].sample(n=min_val)
    #         final_df = pd.concat([final_df, df[df[target_col] == element].sample(n=max_val, )])
        return final_df.reset_index(drop=True)

    return min_idx, min_val, min_class


if __name__=="__main__":

    df_full = pd.read_csv('../../navigant_data/final_df_cl_edit.csv')
    df_full.drop('Unnamed: 0', axis=1, inplace=True)
    targets = ['locationid', 'awo_bucket', 'region', 'npsr', 'awo_amount']
    df_X = df_full.drop(targets, axis=1)
    df_all_y = df_full[targets]

    df_short = df_full.iloc[:100]
    # figure, ax = plt.subplots()
    # pca_mod = PCAModel(df_X.values)
    # pca_mod.make_pca_model(n_components=15)
    # pca_mod.scree_plot(ax, n_components_to_plot=15)

    km = KMeansModel(df_short.values)
    # km.fit()
    # km.elbow_plot(max_clust=5)
    km.silhouette_plot(max_clust=5)

    # vif(x_vals)
    # df_vif = drop_vif_cols(x_vals, 10)
    # df_new = calculate_vif_(x_vals, 10)
