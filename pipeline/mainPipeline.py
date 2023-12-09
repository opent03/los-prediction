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

#import tokenization
#from tokenization import *
import tokenizer2
from tokenizer2 import *

import behrt_train2
from behrt_train2 import *

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

importlib.reload(tokenizer2)
import tokenizer2
from tokenizer2 import *

#importlib.reload(ml_models)
#import ml_models
#from ml_models import *

#importlib.reload(dl_train)
#import dl_train
#from dl_train import *

importlib.reload(behrt_train2)
import behrt_train2
from behrt_train2 import *

importlib.reload(fairness)
import fairness

importlib.reload(callibrate_output)
import callibrate_output

importlib.reload(evaluation)
import evaluation

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
"""
if data_icu:
    token=tokenizer2.BEHRT_models(data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,False)
    tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds, meds_labels, n_meds =token.tokenize()
else:
    token=tokenization.BEHRT_models(data_icu,diag_flag,proc_flag,False,False,med_flag,lab_flag)
    tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels=token.tokenize()
#"""
#behrt_train.train_behrt(tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds, meds_labels, n_meds)
#data2_dir = '/datasets/MIMIC-IV/' + 'data/token/'
#df_split_loc = None
data2_dir = './data/data_random_upsample/features/'
df_split_loc = './data/data_random_upsample/labels_split.csv'
n_meds = 270 #291 for all data 
names = ['tokenized_src', 'tokenized_age', 'tokenized_gender', 'tokenized_ethni', 'tokenized_ins', 'tokenized_labels', 'labs', 'meds']
med_label = np.load(data2_dir+'meds_labels_5000.npy', allow_pickle=True)
all_df = []
for i in range(len(names)):
    df = pd.read_csv(data2_dir+names[i]+'_5000.csv',index_col=0)
    all_df.append(df)
tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds = all_df    

bert_model = behrt_train2.train_behrt(tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, labs, meds, med_label, n_meds, df_split_loc=df_split_loc)
#bert_model.training_phase()
