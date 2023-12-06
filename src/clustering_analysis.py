import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans

class ClusteringEstimator(BaseEstimator):

    def __init__(self, kmax=5, verbose=False):
        self.__dict__.update(locals())

        self.fit_called_ = False
        self.masks_ = None


    def fit(self, subgroup_feature_data, samplewise_losses, feature_names=None):

        models = []
        cluster_inds = []
        worst_perf = []
        masks = []
        def iteration(k):
            model = KMeans(n_clusters=k, n_init=10)
            preds = model.fit_predict(subgroup_feature_data)
            c_losses = np.zeros(k)
            for i in range(k):
                c_losses[i] = samplewise_losses[model.labels_ == i].mean()
            c_idx = np.argmax(c_losses)
            worst_perf.append(max(c_losses))
            cluster_inds.append(preds)
            models.append(model)
            masks.append(model.labels_ == c_idx)
        
        if self.verbose:
            for k in tqdm(range(2, self.kmax+1)):
                iteration(k)
        else:
            
            for k in range(2, self.kmax+1):
                iteration(k)
                
        idx = np.argmax(worst_perf)
        self.model_ = models[idx]
        self.masks_ = masks[idx]
        self.fit_called_ = True