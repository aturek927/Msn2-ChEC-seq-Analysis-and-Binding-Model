#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:15:04 2026

@author: alex-turek
"""

import os
import lightgbm as lgb
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import regex
import dill
import gzip
import toolz, itertools, more_itertools
from collections import Counter, OrderedDict
import optuna
import sklearn
import math
import pandas as pd 
import numpy as np
import seaborn as sns
from imblearn.combine import SMOTETomek
import statistics
import shap
np.random.seed(42)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from matplotlib import pyplot
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import RepeatedStratifiedKFold
import collections

DIR = Path('/home/alexander-turek/Documents/DNA-structure-prediction-main')
assert DIR.exists()
DATA = DIR/"data"

package_root = Path(__file__).parent

def make_dirs(*, name: str) -> ():
    try:
        _file = DIR / "data"
        os.mkdir(_file / name)
    except FileExistsError:
        print("Dir exists")


def pkl_it(dataframe, filebase):
    with open(DIR / "data" / "pkl" / str(filebase + ".pkl"), "wb") as fh:
        dill.dump(dataframe, fh)
    return


def unpkl_it(filebase):
    file_name = str(filebase + ".pkl")
    file_loc = package_root/file_name
    with open(file_loc, "rb") as fh:
        return dill.load(fh)

import logging
from collections import defaultdict
logging.basicConfig(filename=DATA/'results.log', level=logging.INFO)

# A flexible helper Class for running different ML algorithms
# It automatically chooses the best threshold for classification by locating the arg_max (index) of the best F-score

class Call_Plot():
    
    def __init__(self, sklearn_model=False, model_name="LGBM", repeated_k_fold=False):
        plt.close()
        self.model_name = model_name
        self.fig, self.ax = plt.subplots()
        self.ax.plot([0,1], [0,1], linestyle='--', label='Random choice')
        self.ax.set_xlabel('False Positive Rate', fontsize=12)
        self.ax.set_ylabel('True Positive Rate', fontsize=12)
        
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlabel('Recall', fontsize=12)
        self.ax2.set_ylabel('Precision', fontsize=12)

        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.no_skill = []
        self.sklearn_model = sklearn_model
        self.results = defaultdict(list)
        self.repeated_k_fold = repeated_k_fold
        
        
    def Plot(self, data: dict, model, idx):
        if self.sklearn_model:
            y_pred_val = model.predict_proba(data["X_val"])[:,1]
        else:
            y_pred_val = model.predict(data["X_val"])
        
        #Precision-Recall
        precision, recall, thresholds = precision_recall_curve(data["y_val"], y_pred_val)
        no_skill = len(data["y_val"][data["y_val"]==1]) / len(data["y_val"])
        self.no_skill.append(no_skill)
        avg_pr = average_precision_score(data["y_val"], y_pred_val)
        auc_pr = sklearn.metrics.auc(recall, precision)
        if self.repeated_k_fold:
            self.ax2.plot(recall, precision, marker='.', label=f'Run {(idx)//5+1} Test Fold{(idx)%5+1}: AUC PR={auc_pr:.2f}')
        else:
            self.ax2.plot(recall, precision, marker='.', label=f'Test Fold{(idx)+1}: AUC PR={auc_pr:.2f}')
        
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix_pr = np.argmax(fscore)
        self.ax2.scatter(recall[ix_pr], precision[ix_pr], marker='o', color='black')
        
        Accuracy = sklearn.metrics.accuracy_score(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        target_names = ['B-DNA', 'A-DNA']
        print(classification_report(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0), target_names=target_names))
        F1 = sklearn.metrics.f1_score(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        MCC = sklearn.metrics.matthews_corrcoef(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        cohen_kappa_score = sklearn.metrics.cohen_kappa_score(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        logging.info(f'Fold {idx + 1}: Average PR: {avg_pr:.2f} ')
        logging.info(f'Fold {idx + 1}: AUC PR: {auc_pr:.2f} ')
        logging.info(f'Fold {idx + 1}: Best Threshold_f-score={thresholds[ix_pr]:.2f}, F-Score={fscore[ix_pr]}')
        logging.info(f'Fold {idx + 1}: Accuracy: {Accuracy:.2f}')
        logging.info(f'Fold {idx + 1}: F1: {F1:.2f}')
        logging.info(f'Fold {idx + 1}: MCC: {MCC:.2f}')
        
        #ROC-AUC
        fpr, tpr, thresholds_auc = roc_curve(data["y_val"], y_pred_val)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        if self.repeated_k_fold:
            self.ax.plot(fpr, tpr, marker='.',
                     label=f'Run {(idx)//5+1} Test Fold{(idx)%5+1}: AUC={sklearn.metrics.auc(fpr, tpr):.2f}')
        else:
            self.ax.plot(fpr, tpr, marker='.',
                     label=f'Test Fold{(idx)+1}: AUC={sklearn.metrics.auc(fpr, tpr):.2f}')
        self.ax.scatter(fpr[ix], tpr[ix], marker='o', color='black')
        # axis labels
        self.ax.legend(loc="lower left")
        # Mean plot
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(gmeans[ix])
        
        logging.info(f'Fold {idx + 1}: ROC-AUC: {sklearn.metrics.auc(fpr, tpr):.2f}')
        logging.info(f'Fold {idx + 1}: Best Threshold_ROC={thresholds_auc[ix]:.2f}, G-Mean_ROC={gmeans[ix]:.2f}')
        print("Average PR: ", avg_pr )
        print("AUC PR: ", auc_pr)
        print('Best Threshold_f-score=%f, F-Score=%.3f' % (thresholds[ix_pr], fscore[ix_pr]))
        print("AUC: ", sklearn.metrics.auc(fpr, tpr))
        print('Best Threshold_ROC=%f, G-Mean_ROC=%.3f' % (thresholds_auc[ix], gmeans[ix]))
        print("Accuracy: ", Accuracy )
        print("F1: ", F1 )
        print("MCC: ", MCC )
        self.results["Average PR"].append(avg_pr)
        self.results["AUC PR"].append(auc_pr)
        self.results["ROC AUC"].append(sklearn.metrics.auc(fpr, tpr))
        self.results["Accuracy"].append(Accuracy)
        self.results["F1"].append(F1)
        self.results["MCC"].append(MCC)
        self.results["cohen_kappa_score"].append(cohen_kappa_score)

    
    def post_Plot(self):
        from sklearn.metrics import auc
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        self.ax.plot(self.mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        self.ax.legend(loc=(0.45, .05),fontsize='medium')
        self.fig.savefig(DIR/"data"/"results"/f"{self.model_name}_AUC_ROC.png", dpi=600)
        no_skill = np.mean(np.array(self.no_skill))
        self.ax2.plot([0,1], [no_skill,no_skill], linestyle='--', label="Random")
        self.ax2.legend(loc=(0.050, .08),fontsize='medium')
        self.fig2.savefig(DIR/"data"/"results"/f"{self.model_name}_AUC_PR.png", dpi=600)

#Class weight factor:
SCALE_POS_WEIGHT = lambda T,P: (100*(T-P)/T) / (100*P/T)

# Need to split the curated dataset into the inputs(x) and outputs(y)
curated_data = unpkl_it("curated_dataset") # unpickle the curated data file
curated_inputs = curated_data.drop(labels=["target", "sequence", "item"], axis=1) # take inputs from curated dataset
curated_outputs = curated_data["target"] # take outputs of from curated dataset
curated_data_dict = {"X_train": curated_inputs, "y_train": curated_outputs} # put inputs and outputs into dictionary to pass to trainer

# train the LightGBM classifier on the curated dataset with optimal hyperparmeter values
# copy the function to train the classifier
def trainer_lgbm_var(data, lambda_l1_val=0.002451474, lambda_l2_val=4.81e-7, num_leaves_val=63, feature_fraction_val=0.494087396,
                     bagging_fraction_val=0.904993524, bagging_freq_val=1, min_child_samples_val=9):
    # by defining the desired hyperparameters with defaults, any of them can be changed by specfying a value for keyword arguement
    # use the best hyperparameters from the set of best lgmb hyperparametrs in Gupta et al. (2021) excel file
    train_x, train_y = data["X_train"], data["y_train"]
    dtrain = lgb.Dataset(train_x, label=train_y, free_raw_data=True)
    param = {
        "objective": "binary",
        "seed": 42,
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "lambda_l1": lambda_l1_val,
        "lambda_l2": lambda_l2_val,
        "num_leaves": num_leaves_val,
        "feature_fraction": feature_fraction_val,
        "bagging_fraction": bagging_fraction_val,
        "bagging_freq": bagging_freq_val,
        "min_child_samples": min_child_samples_val,
        "scale_pos_weight": SCALE_POS_WEIGHT(len(curated_data), curated_data.target.sum()),
    }
    model = lgb.train(param, dtrain,
                    num_boost_round=150
                    )
    return model


def get_model():
    # function to check if a model has already been stored or if one needs to be trained
    # this will help ensure that model only has to be trained once and any other time after that,
    # the model can be just pulled from its file rather retrained upon calling the module
    model_file = package_root/'LightGBM_AB_DNA_Classifier.txt'
    if os.path.exists(model_file):
        trained_model = lgb.Booster(model_file=model_file)
    else:
        print('Model parameter file does not exist. Will train model and create parameter file for future use')
        # train a model if model file is not present
        
        trained_model = trainer_lgbm_var(curated_data_dict)
        trained_model.save_model(model_file)
    return trained_model
trained_model = get_model()

def seq_to_data(dna_sequence): # converts an input DNA sequence into its number of unique dinucleotide steps
    # create a dictionary with the different dinucleotide steps
    dinuc_step_dict = {'AA': 'TT', 'GG':'CC', 'AC':'GT', 'CA':'TG', 'AT':'AT', 'TA':'TA', 'AG':'CT', 'GA':'TC', 'CG':'CG', 'GC':'GC'}
    # create an dictionary to store the number of unique dinucleoptide steps in the sequence
    result_dict = {'AA/TT':0, 'GG/CC':0, 'AC/GT':0, 'CA/TG':0, 'AT/AT':0, 'TA/TA':0, 'AG/CT':0, 'GA/TC':0, 'CG/CG':0, 'GC/GC':0} 
    for i in range(len(dna_sequence)-1): # iterate over each dinucleotide step in the sequence
        step = dna_sequence[i:i+2] # access the dinucleotide step
        if step in tuple(dinuc_step_dict.keys()):
            key = step + '/' + dinuc_step_dict[step]
            result_dict[key] += 1
        elif step in tuple(dinuc_step_dict.values()):
            key = [k for k,v in dinuc_step_dict.items() if v == step][0] + '/' + step
            result_dict[key] += 1
    return result_dict

sym_map = np.zeros(16, dtype=np.uint8)
sym_map[[0,15]] = 0
sym_map[[10,5]] = 1
sym_map[[1,11]] = 2
sym_map[[4,14]] = 3
sym_map[[3]] = 4
sym_map[[12]] = 5
sym_map[[2,7]] = 6
sym_map[[8,13]] = 7
sym_map[[6]] = 8
sym_map[[9]] = 9

def array_to_data(str_array):
    '''
    Function that converts an array into a pandas DataFrame that can be put into the 
    conformation predictor. This was created to vectorize the process of inputting strings
    into the predictor and speed things up.

    Parameters
    ----------
    str_array : ndarray
        Numpy array that is a numerical representation of the strings to be predicted. Each
        row in the array is a single sequence and each column represnts a base. The values
        are as follows: A = 0, C = 1, G = 3, and T = 3. This allows each dinucleotide base
        to be represented by its own number that is deterimied by 4*base1 + base2. 

    Returns
    -------
    data_df : pandas DataFrame
        DataFrame containing the counts of each dinucleotide step in the sequence.

    '''
    dinuc_indices = 4*str_array[:, :-1] + str_array[:, 1:]
    dinuc_map = sym_map[dinuc_indices]
    num_seqs, seq_length = dinuc_map.shape
    offsets = np.arange(num_seqs)[:, None]*10
    shifted = dinuc_map + offsets
    flat_count = np.bincount(shifted.ravel(), minlength=num_seqs*10)
    final_counts = flat_count.reshape(num_seqs, 10)
    return final_counts

def array_to_df(final_counts):
    # convert final_counts array containing all subsequences in the correct input format
    # into a dataframe that is passed to the predictor
    labels = ['AA/TT', 'GG/CC', 'AC/GT', 'CA/TG', 'AT/AT', 'TA/TA', 'AG/CT', 'GA/TC', 'CG/CG', 'GC/GC']
    data_df = pd.DataFrame(data = final_counts, columns = labels)
    return data_df

def data_to_df(data_dict): # convert a dictionary containing the converted sequence data into a panda data frame
    return pd.DataFrame([data_dict])

def AB_form_predictor(data_df): 
    # edit: to calculate the probabilities of many sequences at a time, it is better to keep
    # the model input as a panda DataFrame rather than as a single string
    prob_to_be_A = trained_model.predict(data_df) # get predicted probability of being A-DNA conformation
    return prob_to_be_A


