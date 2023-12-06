#!/home/adarsh.subbaswamy/anaconda3/envs/afisp/bin/python

import argparse
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
import sklearn.metrics as metrics

import sys
sys.path.append('../src')
sys.path.append('slicefinder')
import utils
import sirus
from stability_analysis import LatentSubgroupShiftEstimator
import sirus
import subprocess


def sigmoid(x):
    return(1./(1. + np.exp(-x)))


def generate_data(n=10000, dim=10, features=None, coeffs=None, cardinality=2):
    if cardinality > 2:
        raise NotImplementedError("Currently only supports binary features 'cardinality == 2'")
    X = np.random.randint(low=0, high=cardinality, size=(n, dim))
    # bernoulli to rademacher
    X = 2*X - 1

    if features is None:
        num_relevant_features = max(1, min(np.random.poisson(3), dim))
        features = sorted(np.random.choice(range(dim), num_relevant_features, replace=False))
    if coeffs is None:
        signs = np.random.binomial(1, 0.6, num_relevant_features)
        signs = 2*signs - 1
        # signs *= 2
        coeffs = np.random.randn(num_relevant_features)*0.5**2 + signs
    print(coeffs)
    logit = np.dot(X[:, features], coeffs)
    probs = sigmoid(logit)
    y = np.random.binomial(1, probs)
    
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_samples', dest='num_samples', default=10000, type=int)
    parser.add_argument('--dim', dest='dim', default=10, type=int)

    parser.add_argument('--num_repeats', dest='num_repeats', default=5, type=int)

    args = parser.parse_args()

    N = args.num_samples
    dim = args.dim

    X, y = generate_data(50000, dim=dim)

    model = linear_model.LogisticRegression(C=1e6, fit_intercept=False)
    model.fit(X, y)

    X_test, y_test = generate_data(N, dim)

    # add noise to a subgroup
    # max(X1, -X2) < -2; same as x1 < -2 AND x2 > 2
    subgroup_mask =  (X_test[:, 2] == -1) & (X_test[:, 3] == 1)
    # subgroup_mask = np.maximum(X_test[:, 1], -X_test[:, 2]) < -2
    flipped_inds = []
    for i in range(len(y_test)):
        if subgroup_mask[i]:
            if np.random.binomial(1, 0.5) == 1:
                if y_test[i] == 1:
                    y_test[i] = 0
                else:
                    y_test[i] = 1
                flipped_inds.append(i)
                
    # y_test[subgroup_mask] = np.random.binomial(1, .5, sum(subgroup_mask))


    test_preds = model.predict_proba(X_test)[:, 1]
    loss_fn = utils.brier
    test_loss = loss_fn(y_test, test_preds)
    n_repeats = args.num_repeats
    runtimes = []
    out_file_name = f"out/afisp_runtimes_{N}_{dim}_{n_repeats}"

    for iteration in range(n_repeats):
        start = time.time()
        stability_analysis = LatentSubgroupShiftEstimator(cv=5,
                                                 verbose=False,
                                                 eps=1e-5)

        sa_risks = stability_analysis.fit(X_test, test_loss)

        sa_masks = stability_analysis.subset_masks()
        phenotype_df = pd.DataFrame(X_test)
        phenotype_df = phenotype_df.add_prefix('X')

        phenotype_df['subset_label'] = sa_masks[1]*1
        time_str = time.strftime("%Y%m%d-%H%M%S")
        df_fname = f"tmp/afisp_simulated_phenotype_df_{N}_{dim}_{iteration}_{time_str}.csv"
        phenotype_df.to_csv(df_fname, index=False)
        sirus_rules_fname = f"tmp/simulated_sirus_rules_{N}_{dim}_{iteration}_{time_str}.txt"
        subprocess.call(f"/home/adarsh.subbaswamy/anaconda3/envs/afisp/bin/Rscript run_sirus.r --input {df_fname} --output {sirus_rules_fname}",
               shell=True)
        sirus_rules = sirus.get_sirus_rules(sirus_rules_fname)
        rule_p_values = sirus.precompute_p_values(sirus_rules, phenotype_df, test_loss)
        significant_rules = sirus.holm_bonferroni_correction(rule_p_values)

        extracted_rules = sirus.effect_size_filtering(significant_rules, phenotype_df, test_loss)

        end = time.time()
        elapsed = end - start
        runtimes.append(elapsed)
    np.savetxt(f"out/afisp_runtime_{N}_{dim}_{n_repeats}.out", np.array(runtimes))