###################################################################################################
# File: RunDiamond.py
# Version: 0.0
# Date: 6.12.2017
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
# import matplotlib.pyplot as plt
# from stats.HigherLevelRanksum import directed_mannwhitneyu
# from scipy.stats.stats import spearmanr, pearsonr
# from scipy.stats import ttest_ind, kstest, ks_2samp
# from scipy.spatial.distance import braycurtis, pdist, squareform, euclidean, cdist
import seaborn as sns
from datetime import datetime

from queue.qp import qp
import argparse
sns.set_style("darkgrid")

DIAMOND_COMNNAD = '/home/noamba/Genie/Bin/diamond/diamond'
GENES_KOS_DICT = '/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/KEGG_DB_12_12_2017/genes_kos_dict.dat'

def run_diamond(command_args, f, force_write = False):
    more_sensitive = ''
    if command_args.more_sensitive:
        more_sensitive = '--more-sensitive'
        
    outputdir = command_args.reference.split('/')[-1].split('.')
    if len(outputdir) > 1:
        outputdir = command_args.output_dir + '/' + ''.join(outputdir[:-1])
    else:
        outputdir = command_args.output_dir + '/' + outputdir[0]
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        
    output_path = '/'.join([outputdir, f.split('/')[-1].split('.fastq')[0] + '.diam'])
    if os.path.exists(output_path):
        if force_write:
            print output_path + ' already exists - force writing.'
        else:
            print output_path + ' already exists'
            return
        
    diamond_command = ' '.join([DIAMOND_COMNNAD, command_args.diamond_mapper, '-d', 
                                command_args.reference, '-q', f, '-o', 
                                output_path, 
                                '--max-target-seqs', str(command_args.max_target_seqs), 
                                '--evalue', str(command_args.evalue), more_sensitive])
    os.system(diamond_command)
    return
# diamond blastx -d K20038_protein -q ../Cardio/Cardio07112017/tmp/Raw/SSUnitedFastq/FD2481_1390.fastq -o matches.m8 --max-target-seqs 1 --evalue 10 --more-sensitive


def upload_run_diamond_jobs_to_q(q, command_args):
    pref_to_use = None
    if command_args.take_only_from_file is not None:
        if os.path.exists(command_args.take_only_from_file):
            pref_to_use = Utils.Load(command_args.take_only_from_file)
    waiton = []
    fastq_files_to_run = []
    for d in command_args.list_of_directories.split('---'):
        print d
        for f in os.listdir(d):
            if re.match(".*fastq$", f):
                if pref_to_use is not None:
                    if f.split('.fastq')[0] in pref_to_use:
                        pref_to_use.remove(f.split('.fastq')[0])
                        fastq_files_to_run.append('/'.join([d, f]))
        print len(fastq_files_to_run)
    
    print "Will run over " + str(len(fastq_files_to_run)) + ' fastq files.'
    for f in fastq_files_to_run:
        waiton.append(q.method(run_diamond, 
                               (command_args, 
                                f)))

    res = q.waitforresults(waiton)
    return res

def parse_diamond_output(command_args):
    # this is a mess
    # It should be much more generic
    border_vals = [10./10**(s) for s in range(15)]
    border_vals.reverse()
    outputdir = command_args.reference.split('/')[-1].split('.')
    if len(outputdir) > 1:
        outputdir = command_args.output_dir + '/' + ''.join(outputdir[:-1])
    else:
        outputdir = command_args.output_dir + '/' + outputdir[0]
    counts_df = pd.DataFrame(columns=border_vals)
    genes_kos_dict = None
    if command_args.divide_to_ko:
        genes_kos_dict = Utils.Load(GENES_KOS_DICT)
        counts_df = pd.DataFrame()

    for f in os.listdir(outputdir):
        f_names = f.split('.dia')
        if len(f_names) != 2:
            continue
        print f
        samp_name = f_names[0]
        if command_args.divide_to_ko:
            counts_df = count_lines_divide_to_ko(command_args, counts_df, outputdir + '/' + f, samp_name, genes_kos_dict)
        else:
            counts_df.loc[samp_name, :] = count_lines_by_value(border_vals, outputdir + '/' + f, is_lower=True)
#         counts_df.loc[samp_name, 'n_reads'] = sum(1 for line in open(outputdir + '/' + f))
    print "writing to file"
    if command_args.divide_to_ko:
        Utils.Write(outputdir + '/counts_divide_by_ko_' + str(command_args.divide_to_ko_th) + '.dat', counts_df)
    else:
        Utils.Write(outputdir + '/counts.dat', counts_df)
    return

def count_lines_divide_to_ko(command_args, df, f, samp_name, genes_kos_dict):
    if df.shape[0] != 0:
        df.loc[samp_name] = 0
    for line in open(f, 'r'):
        fields = line.split()
        kos = genes_kos_dict[fields[1]]
        for ko in kos:
            if ko not in df.columns:
                df[ko] = np.zeros((df.shape[0]))
                df.loc[samp_name, ko] = 0
            if float(fields[10]) < command_args.divide_to_ko_th:
                df.loc[samp_name, ko] = df.loc[samp_name, ko] + 1
    return df

def count_lines_by_value_divide_to_ko():
    # multi index 
    # fd_file = 'DIAMOND/Yeela_pathways_and_kos_17122017_protein/FD2522_1458.diam'
    # border_vals = [10./10**(s) for s in range(15)]
    # border_vals.reverse()
    # counts_df = pd.DataFrame(index=pd.MultiIndex(levels=[[], []], labels=[[], []], names=['sample', 'KO']), columns=border_vals)
    
    
    # for line in open(fd_file, 'r'):
    #     fields = line.split()
    #     kos = genes_kos_dict[fields[1]]
    #     for ko in kos:
    #         if fd_file in counts_df.index.get_level_values(0) and (fd_file, ko) in counts_df.index:
    #             vals = counts_df.loc[(fd_file, ko)].values
    #             pos = find_pos(border_vals, float(fields[10]))
    #             vals[pos:] = vals[pos:] + 1
    #             counts_df.loc[(fd_file, ko), :] = vals
    #         else:
    #             vals = np.zeros((counts.shape[1]))
    #             pos = find_pos(border_vals, float(fields[10]))
    #             vals[pos:] = vals[pos:] + 1
    #             counts_df.loc[(fd_file, ko), :] = vals
    return

def count_lines_by_value(border_vals, f, is_lower = True):
#     for each border vals count the number of lines above or beighth it
    counts = np.zeros((len(border_vals)))
    evals = [float(val.split('\t')[10]) for val in open(f)]
    evals.sort()
    v = 0
    i = 0
    # go over sorted list and find pivots, that way perform the counts
    while  i < len(evals):
        if border_vals[v] <= evals[i]:
            if v == len(border_vals)-1:
                break
            counts[v+1] = counts[v]
            v += 1
        else:
            counts[v] += 1
            i += 1
    if v < len(border_vals)-1:
        counts[v:] = counts[v]
    return counts




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('list_of_directories', help='List of directories to take mapping files from separated with ---', type=str, default=None)
    parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    parser.add_argument('-reference', help='Path to DIAMOND reference file', type=str, default=None)
    parser.add_argument('-max_target_seqs', help='--max-target-seqs parameter in DIAMOND', type=int, default=1)
    parser.add_argument('-evalue', help='--evalue parameter in DIAMOND', type=float, default=10.)
    parser.add_argument('-more_sensitive', help='--more-sensitive parameter in DIAMOND', type=bool, default=True)
    parser.add_argument('-take_only_from_file', help='Path to list of samples prefixes to take', type=str, default=None)
    parser.add_argument('-diamond_mapper', help='Which diamond mapper to use, one of blastx, blastp', type=str, default='blastx')
    parser.add_argument('-only_parse', help='Whether to only parse and create results', type=bool, default=False)
    parser.add_argument('-divide_to_ko', help='Whether to divide the mapping counts into KOs', type=bool, default=False)
    parser.add_argument('-divide_to_ko_th', help='Threshold for divide to ko', type=float, default=1e-4)

    command_args = parser.parse_args()
    
    if command_args.list_of_directories is None or command_args.output_dir is None:
        return
    if command_args.only_parse:
        parse_diamond_output(command_args)
        return
    dirs_list = command_args.list_of_directories.split('---')
    for d in dirs_list:
        if not os.path.exists(d):
            print d + ' does not exists.'
            return
    if not os.path.exists(command_args.output_dir):
        os.makedirs(command_args.output_dir)

    if command_args.max_target_seqs < 1:
        return
    
    if command_args.take_only_from_file is not None:
        if not os.path.exists(command_args.take_only_from_file):
            return
    
    with open(command_args.output_dir + '/args' + str(datetime.now()), 'w') as handle:
        for arg in vars(command_args):
            handle.write(str(arg) + '\t' + str(getattr(command_args, arg)) + '\n')
            
    with qp(jobname = 'RDiamond', q=['himem7.q'], mem_def = '5G', trds_def = 1, tryrerun=True, 
        max_u = 310, delay_batch=15) as q:
        os.chdir("/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/temp_q_dir/")
        q.startpermanentrun()
        upload_run_diamond_jobs_to_q(q, command_args)
        
    parse_diamond_output(command_args)
    return

if __name__ == "__main__":
    sethandlers()
    main()