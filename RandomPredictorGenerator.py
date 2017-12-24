###################################################################################################
# File: RandomPredictorGenerator.py
# Noam Bar, noam.bar@weizmann.ac.il
#
# 
# Python version: 2.7
###################################################################################################

from addloglevels import sethandlers
import Utils
import pandas as pd
from pandas import concat, read_csv, Series
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from stats.HigherLevelRanksum import directed_mannwhitneyu
from scipy.stats.stats import spearmanr, pearsonr
from scipy.stats import ttest_ind, kstest, ks_2samp
from scipy.spatial.distance import braycurtis, pdist, squareform, euclidean, cdist
# from Analyses.MantelTest import test as mantel_test
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from graphics.GraphHelper import do_PCoA
from skbio.stats.ordination._principal_coordinate_analysis import PCoA
from datetime import datetime
from skbio.stats.distance import DistanceMatrix
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GroupKFold
from sklearn import metrics
from queue.qp import qp
import argparse
sns.set_style("darkgrid")


N_ESTIMATORS = range(500, 10000, 500)
LEARNING_RATE = [0.1, 0.05, 0.02, 0.015, 0.01, 0.009, 0.008, 0.0075, 0.005, 0.002, 0.001]
MAX_DEPTH = range(2,6)
RANDOM_STATE = range(10)
SUBSAMPLE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_FEATURES = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SAMPLES_SPLIT = range(2,8)
SAMPLES_WEIGHT = range(2,10)
N_PCS = range(50, 200, 5)
K_FOLD = [8,9,10,11,12,13,15,20,25]
PREDICTOR_PARAMS = {'n_estimators':N_ESTIMATORS, 'learning_rate':LEARNING_RATE, 'max_depth':MAX_DEPTH,
                     'random_state':RANDOM_STATE, 'subsample': SUBSAMPLE, 'max_features':MAX_FEATURES,
                      'min_samples_split':MIN_SAMPLES_SPLIT}
OTHER_PARAMS = {'samples_weight': SAMPLES_WEIGHT, 'n_pcs':N_PCS, 'k_fold':K_FOLD}

LEGAL_DIM_REDUCTION_METHODS = ['PCA', 'PCoA', 'None']

def run_predictor(command_args, predictor_params, other_params, n_run):
    if command_args.use_projection:
        _run_predictor_and_project(command_args, predictor_params, other_params, n_run)
        return
        
#     if command_args.dim_red_method == 'PCoA':
#         X = Utils.Load(command_args.path_to_X)
#         X = X[:, :other_params['n_pcs']]
#     elif command_args.dim_red_method == 'PCA':
#         X = read_csv(command_args.path_to_X)
#         X = X.set_index('Unnamed: 0').T
#         X = X.values
#     Y = Utils.Load(command_args.path_to_Y)
    X = Utils.Load(command_args.path_to_X)
    if command_args.dim_red_method == 'None':
        other_params['n_pcs'] = None
    else:
        X = X[:, :other_params['n_pcs']]
    Y = Utils.Load(command_args.path_to_Y)
    
    groups = np.array(range(len(X)))
    group_kfold = GroupKFold(n_splits=other_params['k_fold'])
    
    final_pred = np.array([])
    final_test =  np.array([])
    
    for train_index, test_index in group_kfold.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        est = GradientBoostingClassifier(**predictor_params).fit(X_train, y_train, 
                                                                 sample_weight=(y_train*other_params['samples_weight'])+1)
        vals = est.predict_proba(X_test)
        final_pred = np.concatenate((final_pred, vals[:,1]))
        final_test = np.concatenate((final_test, y_test))
        
    fpr, tpr, thresholds = metrics.roc_curve(final_test+1, final_pred, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)
    predictor_params['auc'] = roc_auc
    predictor_params['dim_red_method'] = command_args.dim_red_method
    predictor_params['X'] = command_args.path_to_X
    predictor_params['project_test_data'] = command_args.use_projection
    predictor_params.update(other_params)
    Utils.Write(command_args.output_dir + '/temp_pred' + n_run + '.dat', predictor_params)
    return

def _run_predictor_and_project(command_args, predictor_params, other_params, n_run):

    X = read_csv(command_args.path_to_X)
    X = X.set_index('Unnamed: 0').T
    X = X.values
    Y = Utils.Load(command_args.path_to_Y)
    
    groups = np.array(range(len(X)))
    group_kfold = GroupKFold(n_splits=other_params['k_fold'])
    
    final_pred = np.array([])
    final_test =  np.array([])
    
    for train_index, test_index in group_kfold.split(X, Y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        if command_args.dim_red_method == 'PCA':
            pca = PCA(n_components=other_params['n_pcs'])
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        est = GradientBoostingClassifier(**predictor_params).fit(X_train, y_train, 
                                                                 sample_weight=(y_train*other_params['samples_weight'])+1)
        vals = est.predict_proba(X_test)
        final_pred = np.concatenate((final_pred, vals[:,1]))
        final_test = np.concatenate((final_test, y_test))
        
    fpr, tpr, thresholds = metrics.roc_curve(final_test+1, final_pred, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)
    predictor_params['auc'] = roc_auc
    predictor_params['dim_red_method'] = command_args.dim_red_method
    predictor_params['X'] = command_args.path_to_X
    predictor_params['project_test_data'] = command_args.use_projection
    predictor_params.update(other_params)
    Utils.Write(command_args.output_dir + '/temp_pred' + n_run + '.dat', predictor_params)
    return



def generate_random_predictors(q, command_args):
    waiton = []
    for i in range(command_args.start_from_index, command_args.start_from_index + command_args.n_runs):
        predictor_params, other_params = choose_random_params(command_args)
        waiton.append(q.method(run_predictor, 
                               (command_args, 
                                predictor_params, 
                                other_params,
                                str(i))))

    res = q.waitforresults(waiton)
    return res

def choose_random_params(command_args):
    predictor_params = {k:random_choose_from_list(PREDICTOR_PARAMS[k]) for k in PREDICTOR_PARAMS}
    other_params = {k:random_choose_from_list(OTHER_PARAMS[k]) for k in OTHER_PARAMS}
    return predictor_params, other_params

def random_choose_from_list(l):
    return l[random.sample(range(len(l)),1)[0]]

def concat_temp_files(command_args):
    if os.path.exists(command_args.output_df):
        final_df = Utils.Load(command_args.output_df)
    else:
        final_df = pd.DataFrame()
    temp_files = os.listdir(command_args.output_dir)
    temp_files = [f for f in temp_files if re.match('temp_pred.+', f)]
    num_files = len(temp_files)
    print "Will concat " + str(num_files) + " files"
    i = 0
    for t in temp_files:
        if i % 10 == 0:
            print str(datetime.now()) + ' - ' + str(i)
        i += 1
        dic = Utils.Load(command_args.output_dir + '/' + t)
        final_df = concat((final_df, pd.DataFrame(dic, index=[final_df.shape[0]])))
        os.remove(command_args.output_dir + '/' + t)
    Utils.Write(command_args.output_df, final_df)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    parser.add_argument('-dim_red_method', help='Either PCA, PCoA or None currently only these are supported', type=str, default='PCA')
    parser.add_argument('-output_df', help='Path to final data frame', type=str, default=None)
    parser.add_argument('-n_runs', help='Number of random predictions to run', type=int, default=10)
    parser.add_argument('-start_from_index', help='Start naming files from this index', type=int, default=0)
    parser.add_argument('-path_to_X', help='Path to features data - X', type=str, default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Cardio/Cardio07112017/EMGenes_binary_joined.csv')
    parser.add_argument('-path_to_Y', help='Path to labels - Y', type=str, default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Cardio/Cardio07112017/PCoA_sites_EMGenes_Y.dat')
    parser.add_argument('-only_concat', help='Path to final data frame', type=bool, default=False)
    parser.add_argument('-use_projection', help='Whether to project the test data in each k fold', type=bool, default=False)
    command_args = parser.parse_args()
    
    if command_args.output_dir is None:
        return
    if command_args.output_df is None:
        command_args.output_df = command_args.output_dir + '/final_predictions.dat'
    if (not os.path.exists(command_args.path_to_X)) or (not os.path.exists(command_args.path_to_Y)):
        return
    if command_args.n_runs < 1 or command_args.n_runs > 100000:
        return
    if command_args.only_concat:
        concat_temp_files(command_args)
        return
    if command_args.dim_red_method not in LEGAL_DIM_REDUCTION_METHODS:
        print "dim_red_method must be one of: " + ', '.join(LEGAL_DIM_REDUCTION_METHODS)
        return
#     if (command_args.dim_red_method == 'PCoA') and (not re.match(command_args.dim_red_method + '.+', 
#                                                                  command_args.path_to_X.split('/')[-1])):
#         print "dim_red_method doesn't match input X"
#         return
#     if (command_args.dim_red_method == 'PCA') and (re.match('PCoA.+', command_args.path_to_X.split('/')[-1])):
#         print "dim_red_method doesn't match input X"
#         return
#     if (command_args.use_projection is False) and (not re.match(command_args.dim_red_method + '.+', command_args.path_to_X.split('/')[-1])):
#         print "When not using projection X input must start with the dimensionality reduction method"
#         return
#     if command_args.dim_red_method == 'PCoA' and command_args.use_projection:
#         print "Projection of test data using PCoA isn't supported at the moment"
#         return
    
    if not os.path.exists(command_args.output_dir):
        os.makedirs(command_args.output_dir)
    with open(command_args.output_dir + '/args' + str(datetime.now()), 'w') as handle:
        for arg in vars(command_args):
            handle.write(str(arg) + '\t' + str(getattr(command_args, arg)) + '\n')
    
    if command_args.use_projection:
        mem_def = '20G'
    else:
        mem_def = '1G'
    with qp(jobname = 'RandPredict', q=['himem7.q'], mem_def = mem_def, trds_def = 1, tryrerun=True, 
            max_u = 110, delay_batch=5) as q:
        os.chdir("/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/temp_q_dir/")
        q.startpermanentrun()
        generate_random_predictors(q, command_args)
    concat_temp_files(command_args)

        
if __name__ == "__main__":
    sethandlers()
    main()