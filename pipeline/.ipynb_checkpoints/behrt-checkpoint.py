import ipywidgets as widgets
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
#data_dir = '/datasets/MIMIC-IV/physionet.org/files'
data_dir='~/physionet.org/files'
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


import model.ml_models as ml_models
from ml_models import *

import dl_train
from dl_train import *

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

importlib.reload(ml_models)
import ml_models
from ml_models import *

importlib.reload(dl_train)
import dl_train
from dl_train import *

importlib.reload(behrt_train)
import behrt_train
from behrt_train import *

importlib.reload(fairness)
import fairness

importlib.reload(callibrate_output)
import callibrate_output

importlib.reload(evaluation)
import evaluation


data_icu=True
diag_flag=True
out_flag=True
chart_flag=True
proc_flag=True
med_flag=True

tokenized_src = pd.read_csv('token/tokenized_src_5000.csv', index_col=0)
tokenized_age = pd.read_csv('token/tokenized_age_5000.csv', index_col=0)
tokenized_gender = pd.read_csv('token/tokenized_gender_5000.csv', index_col=0)
tokenized_ethni = pd.read_csv('token/tokenized_ethni_5000.csv', index_col=0)
tokenized_ins = pd.read_csv('token/tokenized_ins_5000.csv', index_col=0)
tokenized_labels = pd.read_csv('token/tokenized_labels_5000.csv', index_col=0)

behrt_train.train_behrt(tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels)