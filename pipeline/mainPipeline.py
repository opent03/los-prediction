#import ipywidgets as widgets
import sys
from pathlib import Path
import os
import importlib

module_path='preprocessing/day_intervals_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path='utils'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='preprocessing/hosp_module_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)
#print(sys.path)
root_dir = os.path.dirname(os.path.abspath('UserInterface.ipynb'))
data_dir = '/datasets/MIMIC-IV/physionet.org/files'

import day_intervals_cohort
from day_intervals_cohort import *

import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

import data_generation_icu

import data_generation
import evaluation

import feature_selection_hosp
from feature_selection_hosp import *

# import train
# from train import *


#import ml_models
#from ml_models import *

#import dl_train
#from dl_train import *

import tokenization
from tokenization import *


import behrt_train
from behrt_train import *

import feature_selection_icu
from feature_selection_icu import *
import fairness
import callibrate_output


importlib.reload(day_intervals_cohort)
import day_intervals_cohort
from day_intervals_cohort import *

importlib.reload(day_intervals_cohort_v2)
import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

importlib.reload(data_generation_icu)
import data_generation_icu
importlib.reload(data_generation)
import data_generation

importlib.reload(feature_selection_hosp)
import feature_selection_hosp
from feature_selection_hosp import *

importlib.reload(feature_selection_icu)
import feature_selection_icu
from feature_selection_icu import *

importlib.reload(tokenization)
import tokenization
from tokenization import *

#importlib.reload(ml_models)
#import ml_models
#from ml_models import *

#importlib.reload(dl_train)
#import dl_train
#from dl_train import *

importlib.reload(behrt_train)
import behrt_train
from behrt_train import *

importlib.reload(fairness)
import fairness

importlib.reload(callibrate_output)
import callibrate_output

importlib.reload(evaluation)
import evaluation

import numpy as np
import pandas as pd

def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

data_icu = True
diag_flag = True
proc_flag = False
out_flag = False
chart_flag = True
med_flag = True

if data_icu:
    token=tokenization.BEHRT_models(data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,False)
    '''
    tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds, meds_labels, n_meds =token.tokenize()
    names = ['tokenized_src', 'tokenized_age', 'tokenized_gender', 'tokenized_ethni', 'tokenized_ins', 'tokenized_labels', 'labs', 'meds']
    print(n_meds)
    all_df = [tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds]
    data2_dir = '/h/chloexq/los-prediction/pipeline/data/features/'
    for i in range(len(all_df)):
        df = all_df[i]
        print(names[i])
        df.to_csv(data2_dir + names[i]+'_5000.csv')
    np.save(data2_dir + 'meds_labels_5000.npy', np.array(meds_labels, dtype=object), allow_pickle=True)
    assert 0
    '''
    data2_dir = '/h/chloexq/los-prediction/pipeline/data/features/'
    # 260 for random sample
    # 259 for upsample
    n_meds = 259
    names = ['tokenized_src', 'tokenized_age', 'tokenized_gender', 'tokenized_ethni', 'tokenized_ins', 'tokenized_labels', 'labs', 'meds']
    #meds_labels = np.load(data2_dir + 'data/token/'+'meds_labels_5000.npy', allow_pickle=True)
    meds_labels = np.load(data2_dir + 'meds_labels_5000.npy', allow_pickle=True)
    all_df = []
    for i in range(len(names)):
        #df = pd.read_csv(data2_dir + 'data/token/'+names[i]+'_5000.csv', index_col=0)
        df = pd.read_csv(data2_dir + names[i]+'_5000.csv', index_col=0)
        all_df.append(df)
    tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds = all_df
else:
    token=tokenization.BEHRT_models(data_icu,diag_flag,proc_flag,False,False, med_flag, lab_flag)
    tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels=token.tokenize()

behrt_train.train_behrt(tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds, meds_labels, n_meds)
