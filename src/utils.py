import numpy as np
import torch
from sklearn.metrics import roc_auc_score
        
                
def clip_predictions(preds, upper_bound=0.99, lower_bound=0.01):
    """Clip probability predictions to be in (0, 1) (open interval).
     
        Args:
            preds: Numpy array of sample predictions.
            upper_bound: Upper bound of clipped predictions.
            lower_bound: Lower bound of clipped predictions.
        Returns:
            Predictions clipped to be in [lower_bound, upper_bound] interval
    """
     
    new_preds = np.copy(preds)
    one_inds = np.where(preds > upper_bound)[0]
    zero_inds = np.where(preds < lower_bound)[0]
    
    new_preds[one_inds] = np.repeat(upper_bound, one_inds.shape[0])
    new_preds[zero_inds] = np.repeat(lower_bound, zero_inds.shape[0])
    
    return new_preds


# Sample wise loss functions
def cross_entropy(y,y_pred, clip=True):
    """Samplewise cross entropy loss for probabilistic classification
     
        Args:
            y: Array of true binary classification labels.
            y_pred: Array of probability predictions (between 0 and 1)
        Returns:
            Array of samplewise cross entropy losses.
    """
    
    # Note: problems if preds are in {0, 1}
    # Clip predictions before using.
    y_pred = clip_predictions(y_pred)
    return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def entropy(y, y_pred):
    return -np.log(y_pred)

def brier(y, y_pred):
    """Samplewise brier score for probabilistic classification
     
        Args:
            y: Array of true binary classification labels.
            y_pred: Array of probability predictions (between 0 and 1).
        Returns:
            Array of samplewise brier scores.
    """
    
    return (y-y_pred)**2

def weighted_zero_one_loss(y, y_pred, prevalence=0.02):
    w1 = 1./prevalence
    w0 = 1./(1. - prevalence)
    incorrect = (y != y_pred)
    return correct * (w1*y + w0*(1-y))

def zero_one_loss(y, y_pred):
    return 1. * (y != y_pred)
    
def mse(y, y_pred):
    """Samplewise mean squared error for regression
     
        Args:
            y: Array of true regression labels.
            y_pred: Array of regression predictions.
        Returns:
            Array of samplewise mean squared errors.
    """
    
    return (y - y_pred)**2

def logit(p):
    # Map probabilities to the real line
    # Note: requires p to be in (0, 1) exclusive
    clipped = clip_predictions(p)
    return np.log(clipped/(1.-clipped))

def hinge_surrogate(labels, logits):
    positives_term = labels * np.maximum(1.0 - logits, 0)
    negatives_term =  (1.0 - labels) * np.maximum(1.0 + logits, 0)
    
    # for surrogate purposes, this should just be the positive term
    return positives_term + negatives_term

def xent_surrogate(labels, logits):
    softplus_term = np.maximum(-logits, 0.0) + np.log(1.0 + np.exp(-np.abs(logits)))
    
    # for surrogate purposes, this should just be the softplus term
    # because labels are all 1
    
    return logits - labels * logits + softplus_term


def pfohl_torch_roc_auc_surrogate(y, y_pred, surrogate='xent'):
    
    
    y_torch = torch.tensor(y)
    # pfohl used log softmax (so log probabilities
    logits_torch = torch.tensor(np.log(y_pred))
    
    logits_difference_torch = logits_torch.unsqueeze(0) - logits_torch.unsqueeze(1)
    labels_difference_torch = y_torch.unsqueeze(0) - y_torch.unsqueeze(1)
    
    # matrex which is 1 if label y_i != label y_j
    abs_label_difference = torch.abs(labels_difference_torch)
    
    signed_logits_difference_torch = logits_difference_torch * labels_difference_torch
    
    # TODO: make it 'DRY'
    if surrogate == 'xent':
        loss = torch.log(torch.sigmoid(signed_logits_difference_torch))
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
    elif surrogate == 'hinge':
        loss = torch.maximum(torch.zeros(1), torch.ones(1) - signed_logits_difference_torch)
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
        
    return np.array(loss.tolist())
    

def torch_roc_auc_surrogate(y, y_pred, surrogate='xent'):
    """PyTorch computation of a surrogate samplewise AUROC loss.
     
        Args:
            y: Array of true binary classification labels.
            y_pred: Array of probability predictions (between 0 and 1)
            surrogate: String specifying which surrogate loss function to use.
                Default is 'xent'.
                'xent': Cross-entropy surrogate.
                'hinge': Hinge loss surrogate.
        Returns:
            Array of samplewise surrogate AUROC losses.
    """
    
    y_torch = torch.tensor(y)
    logits_torch = torch.tensor(logit(y_pred))
    
    logits_difference_torch = logits_torch.unsqueeze(0) - logits_torch.unsqueeze(1)
    labels_difference_torch = y_torch.unsqueeze(0) - y_torch.unsqueeze(1)
    
    # matrex which is 1 if label y_i != label y_j
    abs_label_difference = torch.abs(labels_difference_torch)
    
    signed_logits_difference_torch = logits_difference_torch * labels_difference_torch
    
    # TODO: make it 'DRY'
    if surrogate == 'xent':
        loss = torch.log(torch.sigmoid(signed_logits_difference_torch))
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
    elif surrogate == 'hinge':
        loss = torch.maximum(torch.zeros(1), torch.ones(1) - signed_logits_difference_torch)
        loss = (abs_label_difference * loss).mean(axis=0) * 0.5
        
    return np.array(loss.tolist())
    


def roc_auc_surrogate(y, y_pred, surrogate='xent'):
    
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    if (np.sum(pos_mask) == 0) or (np.sum(neg_mask) == 0):
        raise Exception("Examples are either all positive or all negative")
                           
    logits = logit(y_pred)
        
    
    logits_difference = np.expand_dims(logits, 0) - np.expand_dims(logits, 1)
    labels_difference = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    
    # if there were weights
    # weights_product = np.expand_dims(weights, 0) * np.expand_dims(weights, 1)
    
    signed_logits_difference = labels_difference * logits_difference
    
    # compute surrogate loss
    if surrogate == 'hinge':
        surr_fn = hinge_surrogate
    elif surrogate == 'xent':
        surr_fn = xent_surrogate
    
    surrogate_loss = surr_fn(np.ones_like(signed_logits_difference), signed_logits_difference)
    # 0 out entries where labels were the same
    proxy_auc_loss = np.abs(labels_difference) * surrogate_loss
    # np.mean(proxy_auc_loss, axis=0)
                             
    
    return proxy_auc_loss

    
    
def bootstrap_ci(y_true, y_pred, n_bootstrap=100, confidence=0.95, loss=roc_auc_score, return_samples=False):
    """Computes non-parametric bootstrap confidence interval for model
    performance.
    
    Args:
        y_true: True target labels.
        y_pred: Target predictions (regression preds, probabilistic preds, 
            or classification preds).
        n_bootstrap: Number of bootstrap resamples to perform.
            Default = 100.
        confidence: The confidence level. Float between 0 and 1.
            Default = 0.95 (i.e., 95% confidence)
        loss: Loss function for computing average model performance. Should
            have signature loss(y_true, y_pred).
    Returns:
        The mean performance, the lower interval, and the upper interval.
    """
    
    n = y_true.shape[0]
        
    upper_p = 100 * (1. - (1. - confidence)/2)
    lower_p = 100 * ((1. - confidence)/2)
    
    aucs = []
    
    def bootstrap_resample_inds():
        return np.array(np.random.choice(range(n), n, replace=True))
    
    for i in range(n_bootstrap):
        inds = np.array(bootstrap_resample_inds())
        resample_true = y_true[inds]
        resample_pred = y_pred[inds]
        
        if loss==roc_auc_score:
            if (resample_true.mean() == 1) or (resample_true.mean() == 0):
                continue
        
        aucs.append(loss(resample_true, resample_pred))
    
    
    lower, upper= np.percentile(aucs, [lower_p, upper_p])
    if return_samples:
        return aucs
    return np.mean(aucs), lower, upper