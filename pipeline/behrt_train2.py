
import pandas as pd
import math
import time

import pickle
import numpy as np

import os
import importlib
import sys

import behrt_model2
from behrt_model2 import *
import model.evaluation as evaluation

from torchmetrics import AUROC
from torchmetrics import AveragePrecision
from torchmetrics import Precision
from torchmetrics import Recall

from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/behrt"):
    os.makedirs("./data/behrt")
if not os.path.exists("./saved_models/checkpoint"):
    os.makedirs("./saved_models/checkpoint")
importlib.reload(behrt_model2)
import behrt_model2
from behrt_model2 import *

#torch.manual_seed(42)
#torch.backends.cudnn.deterministic = True
    
class train_behrt():
    def __init__(self,src, age, sex, ethni, ins, target_data, labs, meds, meds_labels, n_meds):

        train_l = int(len(src)*0.70)
        val_l = int(len(src)*0.1)
        test_l = len(src) - val_l - train_l
        number_output = target_data.shape[1]

        file_config = {
            'model_path': './saved_models/', # where to save model
            'model_name': 'CVDTransformer', # model name
            'file_name': 'log.txt',  # log path
        }
        #create_folder(file_config['model_path'])

        global_params = {
            'max_seq_len': src.shape[1],
            'max_age': age.max().max(),
            'month': 1,
            'age_symbol': None,
            'min_visit': 3,
            'gradient_accumulation_steps': 1
        }

        optim_param = {
            'lr': 3e-5,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        }

        train_params = {
            'batch_size': 16,
            'use_cuda': True,
            'max_len_seq': global_params['max_seq_len'],
            'device': "cuda:0" if torch.cuda.is_available() else "cpu",
            'data_len' : len(target_data),
            'train_data_len' : train_l,
            'val_data_len' : val_l,
            'test_data_len' : test_l,
            'epochs' : 50,
            'action' : 'train'
        }

        model_config = {
            'vocab_size': int(src.max().max() + 1), # number of disease + symbols for word embedding
            'hidden_size': 288, # word embedding and seg embedding hidden size
            'seg_vocab_size': 2, # number of vocab for seg embedding
            'age_vocab_size': int(age.max().max() + 1), # number of vocab for age embedding
            'gender_vocab_size': 2 + 1, # TODO: +1 for missing values
            'ethni_vocab_size': int(ethni.max().max()) + 1,
            'ins_vocab_size': int(ins.max().max()) + 1,
            'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
            'hidden_dropout_prob': 0.2, # dropout rate
            'num_hidden_layers': 6, # number of multi-head attention layers required
            'num_attention_heads': 6, # number of attention heads
            'attention_probs_dropout_prob': 0.2, # multi-head attention dropout rate
            'intermediate_size': 256, # the size of the "intermediate" layer in the transformer encoder
            'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
            'initializer_range': 0.02, # parameter weight initializer range
            'number_output' : number_output,
            'number_meds': n_meds,
        }

        


        train_code = src.values[:train_l]
        val_code = src.values[train_l:train_l + val_l]
        test_code = src.values[train_l + val_l:]

        train_labs = labs.values[:train_l]
        val_labs = labs.values[train_l:train_l + val_l]
        test_labs = labs.values[train_l + val_l:]

        train_meds = meds.values[:train_l]
        val_meds = meds.values[train_l:train_l + val_l]
        test_meds = meds.values[train_l + val_l:]

        train_age = age.values[:train_l]
        val_age = age.values[train_l:train_l + val_l]
        test_age = age.values[train_l + val_l:]

        train_labels = target_data.values[:train_l]
        val_labels = target_data.values[train_l:train_l + val_l]
        test_labels = target_data.values[train_l + val_l:]

        train_gender = sex.values[:train_l]
        val_gender = sex.values[train_l:train_l + val_l]
        test_gender = sex.values[train_l + val_l:]

        train_ethni = ethni.values[:train_l]
        val_ethni = ethni.values[train_l:train_l + val_l]
        test_ethni = ethni.values[train_l + val_l:]

        train_ins = ins.values[:train_l]
        val_ins = ins.values[train_l:train_l + val_l]
        test_ins = ins.values[train_l + val_l:]

        train_meds_labels = meds_labels[:train_l]
        val_meds_labels = meds_labels[train_l:train_l + val_l]
        test_meds_labels = meds_labels[train_l + val_l:]

        train_data = {"code":train_code, "age":train_age, "labels":train_labels, "gender" : train_gender, "ethni" : train_ethni, "ins" : train_ins, "labs": train_labs, "meds": train_meds, "meds_labels": train_meds_labels, "n_meds": n_meds}
        val_data = {"code":val_code, "age":val_age, "labels":val_labels, "gender" : val_gender, "ethni" : val_ethni, "ins" : val_ins, "labs": val_labs, "meds": val_meds, "meds_labels": val_meds_labels, "n_meds": n_meds}
        test_data = {"code":test_code, "age":test_age, "labels":test_labels, "gender" : test_gender, "ethni" : test_ethni, "ins" : test_ins, "labs": test_labs, "meds": test_meds, "meds_labels": test_meds_labels, "n_meds": n_meds}

        conf = BertConfig(model_config)
        behrt = BertForEHRPrediction(conf, model_config['number_output'])

        behrt = behrt.to(train_params['device'])

        #models parameters
        transformer_vars = [i for i in behrt.parameters()]

        #optimizer
        optim_behrt = torch.optim.Adam(transformer_vars, lr=3e-5)

        TrainDset = DataLoader(train_data, max_len=train_params['max_len_seq'], code='code')
        trainload = torch.utils.data.DataLoader(dataset=TrainDset, batch_size=train_params['batch_size'], shuffle=True)

        ValDset = DataLoader(val_data, max_len=train_params['max_len_seq'], code='code')
        valload = torch.utils.data.DataLoader(dataset=ValDset, batch_size=train_params['batch_size'], shuffle=True)

        self.behrt = behrt
        self.optim_behrt = optim_behrt
        self.train_params = train_params
        train_loss, val_loss = self.train(trainload, valload, train_params['device'])

        behrt.load_state_dict(torch.load("./saved_models/checkpoint/behrt", map_location=train_params['device']))
        print("Loading succesfull")

        TestDset = DataLoader(test_data, max_len=train_params['max_len_seq'], code='code')
        testload = torch.utils.data.DataLoader(dataset=TestDset, batch_size=train_params['batch_size'], shuffle=True)
        loss, cost, pred, label = self.eval(testload, True, train_params['device'])

        labels = pd.read_csv("./data/behrt/behrt_labels.csv", header=None)
        preds = pd.read_csv("./data/behrt/behrt_preds.csv", header=None)

        labels=labels.drop(0, axis=1)
        preds=preds.drop(0, axis=1)

        preds = torch.FloatTensor(preds.values)
        sig_pred = torch.sigmoid(preds)
        labels = torch.IntTensor(labels.values)

        threshold=0.8

        auroc = AUROC(pos_label=1)
        print('auroc: ', auroc(sig_pred, labels))

        ap = AveragePrecision(pos_label=1)
        print('average precision: ', ap(sig_pred, labels))

        preds_hard = sig_pred
        preds_hard[sig_pred>=threshold] = 1
        preds_hard[sig_pred<threshold] = 0
        pres = Precision()
        print('precision: ', pres(preds_hard, labels))

        recall = Recall()
        print('recall: ', recall(preds_hard, labels))
        loss_func = evaluation.Loss('cpu',True,True,True,True,True,True,True,True,True,True,True)
        loss_func(sig_pred.cpu(),labels.cpu(),preds.cpu(),False,False)

    def run_epoch(self, e, trainload, device):
        tr_loss = 0
        loss_cls_tr = 0
        loss_dab_tr = 0
        start = time.time()
        self.behrt.train()
        if_dab = False
        dab_w = 1
        print('adding adversarial loss: ',if_dab, dab_w)
        for step, batch in enumerate(trainload):
            self.optim_behrt.zero_grad()
            batch = tuple(t for t in batch)
            input_ids, age_ids, gender_ids, ethni_ids, ins_ids, segment_ids, posi_ids, attMask, labels, labs_ids, meds_ids, meds_labels = batch

            labs_ids = labs_ids.to(device)
            meds_ids = meds_ids.to(device)
            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            gender_ids = gender_ids.to(device)
            ethni_ids = ethni_ids.to(device)
            ins_ids = ins_ids.to(device)
            posi_ids = posi_ids.to(device)
            segment_ids = segment_ids.to(device)
            attMask = attMask.to(device)
            labels = labels.to(device)
            meds_labels = meds_labels.to(device)
            logits, logits_meds = self.behrt(input_ids, labs_ids, age_ids, gender_ids, ethni_ids, ins_ids, segment_ids, posi_ids,
                            attention_mask=attMask, if_dab=if_dab)

            #logits = behrt(input_ids, age_ids, gender_ids, ethni_ids, ins_ids, segment_ids, posi_ids,
            #               attention_mask=attMask)

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            loss_cls_tr += loss.item()
            if if_dab:
                criterion_dab = nn.BCEWithLogitsLoss()
                loss_dab = criterion_dab(logits_meds, meds_labels.float())
                loss += dab_w*loss_dab
                loss_dab_tr += dab_w*loss_dab.item() 

            # print("Loss and loss dab")
            # print(loss, loss_dab)
            loss.backward()

            tr_loss += loss.item()
            if step%500 == 0:
                print(f'loss is {loss.item()} for step {step}')
            self.optim_behrt.step()
            del loss
        cost = time.time() - start
        return tr_loss, cost, loss_cls_tr, loss_dab_tr


    def train(self, trainload, valload, device):
        best_val = math.inf
        early_i = 0 #for early stopping 
        for e in range(self.train_params["epochs"]):
            if(early_i==6):
                break
            print("Epoch n" + str(e))
            train_loss, train_time_cost, loss_cls, loss_dab = self.run_epoch(e, trainload, device)
            print('Finished train')
            val_loss, val_time_cost, pred, label = self.eval(valload, False, device)
            train_loss = train_loss / math.ceil((self.train_params["train_data_len"] / self.train_params['batch_size']))
            val_loss = val_loss / math.ceil((self.train_params["val_data_len"] / self.train_params['batch_size']))
            print('TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
            print('TRAIN loss_cls {} loss_dab {}\n'.format(loss_cls, loss_dab))
            print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost))
            if val_loss < best_val:
                print("** ** * Saving fine - tuned model ** ** * ")
                model_to_save = self.behrt.module if hasattr(self.behrt, 'module') else self.behrt
                self.save_model(model_to_save.state_dict(), './saved_models/checkpoint/behrt')
                best_val = val_loss
                early_i = 0
            else: 
                early_i += 1

            if(e==0):
                break
        return train_loss, val_loss


    #%%

    def eval(self, _valload, saving, device):
        tr_loss = 0
        start = time.time()
        self.behrt.eval()
        if saving:
            with open("./data/behrt/behrt_preds.csv", 'w') as f:
                f.write('')
            with open("./data/behrt/behrt_labels.csv", 'w') as f:
                f.write('')

        for step, batch in enumerate(_valload):
            batch = tuple(t for t in batch)
            input_ids, age_ids, gender_ids, ethni_ids, ins_ids, segment_ids, posi_ids, attMask, labels, labs_ids, meds_ids, _  = batch

            input_ids = input_ids.to(device)
            age_ids = age_ids.to(device)
            gender_ids = gender_ids.to(device)
            ethni_ids = ethni_ids.to(device)
            ins_ids = ins_ids.to(device)
            posi_ids = posi_ids.to(device)
            segment_ids = segment_ids.to(device)
            attMask = attMask.to(device)
            labels = labels.to(device)
            labs_ids = labs_ids.to(device)
            meds_ids = meds_ids.to(device)

            logits, _ = self.behrt(input_ids, labs_ids, age_ids, gender_ids, ethni_ids, ins_ids, segment_ids, posi_ids,
                            attention_mask=attMask)

            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

            if saving:
                with open("./data/behrt/behrt_preds.csv", 'a') as f:
                    pd.DataFrame(logits.detach().cpu().numpy()).to_csv(f, header=False)
                with open("./data/behrt/behrt_labels.csv", 'a') as f:
                    pd.DataFrame(labels.detach().cpu().numpy()).to_csv(f, header=False)

            tr_loss += loss.item()
            del loss

        print("TOTAL LOSS", tr_loss)

        cost = time.time() - start
        return tr_loss, cost, logits, labels

    def save_model(self, _model_dict, file_name):
        torch.save(_model_dict, file_name)
