import imodels
import sklearn
from tqdm import tqdm
import numpy as np
from statsmodels.stats.weightstats import ztest, ttest_ind
from joblib import Parallel, delayed

def negate_simple_rule(rule):
    """Negates a rule string of the form "if ARG [<|>|<=|>=] VAL"

    Args:
        rule: A string of a rule of the form "if ARG [<|>|<=|>=] VAL"
    Returns:
        String in which the condition [>|<|>=|<=] has been negated
    """
    if ">=" in rule:
        return(rule.replace(">=", "<"))
    if "<=" in rule:
        return(rule.replace("<=", ">"))
    if ">" in rule:
        return(rule.replace(">", "<="))
    if "<" in rule:
        return(rule.replace("<", ">="))

def get_sirus_rules(filename):

    with open(filename) as f:
        filelines = [line for line in f]
    
    sirus_rules = set()
    
    for line in filelines:
        if " then " not in line:
            continue
        # every rule reads as 'if RULE then ...'
        end = line.index(" then ")
        rule = line[3:end].strip()
        if_prob = float(line.split('then')[1].split()[0])
        else_prob = float(line.split('else')[1].split()[0])
            
        if if_prob > else_prob:# can take the rule as is
            sirus_rules.add(rule)
        elif '&' in rule: # negate a compound rule
            # ~(X & Y) = ~X or ~Y
            # new_rules = []
            for r in rule.split('&'): 
                # new_rules.append(negate_simple_rule(r.strip()))
                sirus_rules.add(negate_simple_rule(r.strip()))
            # sirus_rules.add(' | '.join(new_rules))
        else: # negate a simple rule
            sirus_rules.add(negate_simple_rule(rule))

    return(sirus_rules)


def bootstrap_performance_samples(y_true, y_pred, n_bootstrap=100, loss_fn=sklearn.metrics.brier_score_loss):
    n = y_true.shape[0]
    perfs = []
    
    def bootstrap_resample_inds():
        return np.array(np.random.choice(range(n), n, replace=True))
    
    for i in range(n_bootstrap):
        inds = np.array(bootstrap_resample_inds())
        resample_true = y_true[inds]
        resample_pred = y_pred[inds]
        

        
        perfs.append(loss_fn(resample_true, resample_pred))
        
    return(perfs)


def parallel_precompute_p_values(sirus_rules, phenotype_df, test_loss, alpha=0.05, n_jobs=1):
    # precompute p_values
    alpha = 0.05
    rule_p_values = []
    
    def one_pval(rule):
        rows = phenotype_df.eval(str(rule))
        pval = ttest_ind(test_loss[rows], 
                                 x2=test_loss[~rows], 
                                 value=0.,
                                 alternative='larger',
                                 usevar='unequal')[1]
        return(rule, pval, rows.sum())
    
    rule_p_values = Parallel(n_jobs=n_jobs)(delayed(one_pval)(rule) for rule in sirus_rules)
    # remove nans and sort by p value
    rule_p_values = sorted([x for x in rule_p_values if not np.isnan(x[1])], key=lambda x: x[1])
    # rule_p_values = sorted(rule_p_values, key=lambda x: x[1])
    return(rule_p_values)

def precompute_p_values(sirus_rules, phenotype_df, test_loss, alpha=0.05):
    # precompute p_values
    alpha = 0.05
    rule_p_values = []
    for rule in tqdm(sirus_rules):
        rows = phenotype_df.eval(str(rule))
            
            
        # two independent sample t test that brier score is larger in group than out of group
        pval = ttest_ind(test_loss[rows], 
                         x2=test_loss[~rows], 
                         value=0.,
                         alternative='larger',
                         usevar='unequal')[1]
        # check for nan
        
        # print(rule, rows.sum())
        # print(f"Mean AUC {np.mean(bootstrap_aucs):.3f} Threshold AUC {performance_threshold:.3f} p value {pval:.2f} ")
        rule_p_values.append((rule, pval, rows.sum()))
    # rule_p_values = sorted(rule_p_values, key=lambda x: x[1])
    # filter out nans
    rule_p_values = sorted([x for x in rule_p_values if not np.isnan(x[1])], key=lambda x: x[1])
    return(rule_p_values)


def holm_bonferroni_correction(rule_p_values, sig_value=0.05):
    # expect p-values to be sorted, smallest to largest
    
    m = len(rule_p_values)
    
    significant_rules = []
    
    for k in range(1, m+1):
        rule_k = rule_p_values[k-1]
        # reject with adaptive significance level
        if rule_k[1] < sig_value / (m + 1 - k):
            significant_rules.append(rule_k)
            continue
        print(k, rule_k, sig_value / (m + 1 - k))
        break

    return(significant_rules)

def cohens_d(c0, c1):
    # this one assumes equal sample sizes?
    # return(effect_size2(c0, c1))
    return(cohen_d(c0, c1))
    return(np.mean(c0) - np.mean(c1)) / (np.sqrt((np.std(c0) ** 2 + np.std(c1) ** 2) / 2))

def cohen_d(x,y):
    # unequal sample sizes, but assume shared population std
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

# slicefinder effect size calculator
def effect_size2(sample_a, sample_b):
    sample_b_mean = np.mean(sample_b)

    # correction
    na = len(sample_a)
    nb = len(sample_b)
    sample_b_var = np.var(sample_b)
    # original
    # sample_b_var = (s**2*(n-1) - np.std(sample_a)**2*(len(sample_a)-1))/(n-len(sample_a)-1)
    if sample_b_var < 0:
        sample_b_var = 0.

    diff = np.mean(sample_a) - sample_b_mean
    diff /= np.sqrt( (np.std(sample_a) + np.sqrt(sample_b_var))/2. )
    return diff


def effect_size_filtering(significant_rules, phenotype_df, test_loss, effect_threshold = 0.4, verbose=False):
    rules = []
    for rule in tqdm(significant_rules):
        r = rule[0]
        rows = phenotype_df.eval(str(r))
        cd = cohens_d(test_loss[rows], test_loss[~rows])
        if verbose:
            print(f"{r} Cohen's d {cd:.2f}")
        if cd > effect_threshold:
            rules.append((r, cd))
    # sort from highest effect size to lowest
    rules = sorted(rules, key=lambda x: x[1], reverse=True)
    rules = [r[0] for r in rules] # drop the effect sizes
    return(rules)

def find_max_effect_size(masks, test_loss):
    cds = []
    p_vals = []
    
    for i in range(len(masks)):
        idxs = masks[i]
        odxs = ~masks[i]
        cds.append(cohen_d(test_loss[idxs], test_loss))

    return(np.argmax(cds), max(cds))