import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


    
class LatentSubgroupShiftEstimator(BaseEstimator):
    """
        Class for calculating the risk (i.e., expected loss) of a model under 
        a worst case covariate shift in the distribution of a set of subgroup
        characteristics.
    """
    
    def __init__(self,
                 subset_fractions=[0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0],
                 conditional_loss_model=None,
                 cv = 10,
                 verbose=False,
                 eps=0.0):
        """
            Instantiate a LatentSubgroupShiftEstimator object
        
            Args:
                subset_fractions: List of subset fractions between 0 and 1.
                    For each subset fraction, the worst-case risk of the model
                    will be computed. For a subset fraction f and a datset of
                    size N, this will produce a worst-case subset of size
                    approximately f*N.
                    
                conditional_loss_model: A supervised learning model with
                    sklearn interface for fitting the expected conditional
                    loss of the model to be evaluated. Default is an
                    ExplainableBoostingRegressor.
                    
                cv: Number of folds for cross validation. Instance wise losses
                    are predicted via cross validation.
                    
                verbose: If True, then prints during calls to self.fit(...).
                    Default 'False'.
                    
                eps: Float >= 0 corresponding to max amount of noise to add to
                    expected conditional loss estimates. It is used to break
                    ties between samples with the same expected losses. It
                    incurs statistical bias of at most eps. Set to small 
                    non-zero values if all subgroup features are discrete.
        """
        
        self.__dict__.update(locals())
        self.fit_called_ = False
        self._masks = None
        # self.fit_mu_ = False
        
    def fit(self, subgroup_feature_data, 
            samplewise_losses, feature_names=None):
        """Estimates worst-case losses under latent subgroup shifts.
     
        Args:
            subgroup_features: ((nsamples, nfeatures), ndarray)
                Numpy array containing the subgroup features
                
            feature_names: ((nfeatures,), List)
                List of the subgroup characterizing features.
            
            samplewise_losses: ((nsamples,), array)
                Array of samplewise losses of the model to evaluate's 
                predictions.
            
        Returns:
            Returns an array of worst-case risk estimates, one per subset
            fraction.
        """
        
        # by default we will use Explainable Boosting Machines (EBMs) since
        # we have observed that these models work well out of the box, without
        # any hyperparameter tuning.
        self.mu_mdl = self.conditional_loss_model
        if self.conditional_loss_model is None:
            self.mu_mdl = ExplainableBoostingRegressor(
                feature_names=feature_names
            )
        
        X = subgroup_feature_data
        N = X.shape[0]
        self.num_samples_ = N
        num_subsets = len(self.subset_fractions)
        
        # random noise for when subgroup feature data is all discrete
        U = np.random.rand(N,1)*self.eps
        
        # mu hat will contain our expected conditional loss estimates
        # one per sample
        self.mu_hat_ = np.zeros(N)
        
        # eta hat will contain the sample-specific thresholds corresponding
        # to quantiles of the expected conditional loss. Each sample will
        # have a threshold per subset fraction. The quantile is equal to
        # 1 - the sample fraction.
        self.eta_hat_ = np.zeros((num_subsets,N))
        
        # intermediate computation
        self.h_hat_ = np.zeros((num_subsets,N))
        
        # stores the expected conditional loss model for each fold
        self.mu_mdls_ = []
        
        k = 0
        folds = KFold(self.cv, shuffle=False).split(subgroup_feature_data)
        for train_idxs,test_idxs in folds:
            if self.verbose: print(f"k = {k}")
            k += 1
            
            # fit model to fold                        
            mu_mdl_k = clone(self.mu_mdl).fit(X[train_idxs],
                                              samplewise_losses[train_idxs])
            self.mu_hat_[test_idxs] = mu_mdl_k.predict(X[test_idxs])
            self.mu_mdls_.append(mu_mdl_k)
                        
            # Note: for a given alpha value, the eta value for samples will
            # vary depending on which fold the sample is in
            for a, alpha in enumerate(self.subset_fractions):
                # compute the 1-alpha quantile of the expected cond loss
                # this is the eta value for all samples in this fold
                self.eta_hat_[a, test_idxs] = np.repeat(
                            np.quantile(self.mu_hat_[test_idxs], 1.-alpha), 
                            len(test_idxs)
                )
            
        
        
        # R hat will contain the worst case average loss for each subset
        self.R_hats_ = np.zeros(num_subsets)
        
        # variance estimates and confidence intervals for worst case
        # average loss
        self.sigma_hats_ = np.zeros(num_subsets)
        self.cis_ = np.zeros((num_subsets,2))
            
        for a, alpha in enumerate(self.subset_fractions):
            self.h_hat_[a] = 1.0*(self.mu_hat_ + U[:,0] >= self.eta_hat_[a])
        
            psi = np.maximum(self.mu_hat_ + U[:,0] - self.eta_hat_[a],0.0)/alpha + self.eta_hat_[a]
            psi += self.h_hat_[a]*(samplewise_losses - self.mu_hat_)/alpha
    
            self.R_hats_[a] = np.mean(psi)
            self.sigma_hats_[a] = np.sqrt(np.mean((psi - self.R_hats_[a])**2))
            
            # lower confidence interval
            self.cis_[a,0] = self.R_hats_[a] - 1.96*self.sigma_hats_[a]/np.sqrt(N)
            # upper confidence interval
            self.cis_[a,1] = self.R_hats_[a] + 1.96*self.sigma_hats_[a]/np.sqrt(N)
        
        self.fit_called_ = True
        return self.R_hats_
    
    def confidence_intervals(self):
        """Computes analytical confidence intervals for the worst-case risk
        estimates for each subset fraction size.
        
        Args:
        
        Returns: ((# of subset fraction sizes, 2),ndarray)
            Array with first column containing lower confidence interval and
            second column containing upper confidence interval.
        """
        
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on LatentSubgroupShiftEstimator object first.')
        return self.cis_
    
    def subset_masks(self):
        """Computes a boolean mask for indexing worst-case data subsets.
     
        Args:
            
        Returns:
            Array of boolean masks for indexing the worst-case subset
            corresponding to each subset fraction
        """
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on LatentSubgroupShiftEstimator object first.')
        # find the samples for which expected conditional loss is greater
        # than the quantile.
        # only compute once
        if self._masks is None:
            self._masks = [self.mu_hat_ >= self.eta_hat_[a] 
                for a in range(len(self.subset_fractions))]
        
        return self._masks
    
    def check_subset_sizes(self, plot=True, ax=None):
        """Diagnostic for checking that fitting risk estimator worked.
     
        Args:
            plot: bool, default=True
                Boolean flag for plotting the subset size checking diagnostic.
                
            ax: matplotlib axis, default=None
                Matplotlib axis to plot on.
            
        Returns:
            Two lists, the first containing the subset fractions and the second
            containing the extracted fractions. When plotted against each other
            the relationship should match y=x.
        """
        
        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on LatentSubgroupShiftEstimator object first.')
        observed_fractions = [np.mean(m) for m in self.subset_masks()]
        
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(self.subset_fractions, observed_fractions)
            ax.plot([0,1],[0,1],'k:', label='Perfect fit')
            ax.set_xlabel('Subset Fraction')
            ax.set_ylabel('Fraction Selected by Worst-Case Mask')
            ax.legend(loc='best')

        
        return self.subset_fractions, observed_fractions