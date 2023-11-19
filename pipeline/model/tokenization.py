import pandas as pd
import pickle
import numpy as np
import tqdm
import os
import importlib
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

class BEHRT_models():
    def __init__(self,data_icu,diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag):
        self.data_icu=data_icu
        if self.data_icu:
            self.id='stay_id'
        else:
            self.id='hadm_id'
        self.diag_flag,self.proc_flag,self.out_flag,self.chart_flag,self.med_flag,self.lab_flag=diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag
        
    def tokenize_dataset(self,labs_input, cond_input, demo_input, labels, vocab, demo_vocab, ins_vocab, gender_vocab, labs_tokens, meds_tokens):
        tokenized_src = []
        tokenized_gender = []
        tokenized_ethni = []
        tokenized_ins = []
        tokenized_age = []
        tokenized_labels = []

        tokenized_labs = []
        tokenized_meds = []

        meds_labels = []

        idx = 0

        print("STARTING TOKENIZATION.")

        for patient, group in tqdm.tqdm(labs_input.groupby(self.id)):
            tokenized_src.append([])
            tokenized_src[idx].append(vocab["token2idx"]['CLS'])
            
            tokenized_labs.append([])
            tokenized_labs[idx].append(vocab["token2idx"]['CLS'])

            tokenized_meds.append([])
            tokenized_meds[idx].append(vocab["token2idx"]['CLS'])

            meds_labels.append([])

            for row in cond_input[cond_input[self.id] == patient].itertuples(index=None):
                for key, value in row._asdict().items():
                    if value == '1':
                        tokenized_src[idx].append(vocab["token2idx"][key])
            tokenized_src[idx].append(vocab["token2idx"]['SEP'])

            for lab in group.itertuples(index=None):
                for col in lab:
                    if not isinstance(col, float) and not pd.isnull(col):
                        tokenized_src[idx].append(vocab["token2idx"][col])
                        if col in labs_tokens:
                            tokenized_labs[idx].append(vocab["token2idx"][col])
                        if col in meds_tokens:
                            tokenized_meds[idx].append(vocab["token2idx"][col])
                            if col not in meds_labels[idx]:
                                meds_labels[idx].append(col)
                tokenized_src[idx].append(vocab["token2idx"]['SEP'])
                tokenized_labs[idx].append(vocab["token2idx"]['SEP'])
                tokenized_meds[idx].append(vocab["token2idx"]['SEP'])
            
            tokenized_src[idx][-1] = vocab["token2idx"]['SEP']
            tokenized_labs[idx][-1] = vocab["token2idx"]['SEP']
            tokenized_meds[idx][-1] = vocab["token2idx"]['SEP']

            if len(tokenized_src[idx]) >= 512:
                # TODO: Random sample tokens between CLS and SEPs instead of truncation
                tokenized_src[idx] = tokenized_src[idx][:512]
                tokenized_labs[idx] = tokenized_labs[idx][:512]
                tokenized_meds[idx] = tokenized_meds[idx][:512]
                meds_labels[idx] = meds_labels[idx][:512]
                # tokenized_src.pop()
            if len(meds_labels[idx]) == 0:
                meds_labels[idx] = ["nan"]
            #else:
            gender = gender_vocab[demo_input[demo_input[self.id] == patient].iloc[0, 1]]
            ethnicity = demo_vocab[demo_input[demo_input[self.id] == patient].iloc[0, 2]]
            insurance = ins_vocab[demo_input[demo_input[self.id] == patient].iloc[0, 3]]
            age = demo_input[demo_input[self.id] == patient].iloc[0, 0]
            tokenized_gender.append([gender] * len(tokenized_src[idx]))
            tokenized_ethni.append([ethnicity] * len(tokenized_src[idx]))
            tokenized_ins.append([insurance] * len(tokenized_src[idx]))
            tokenized_age.append([age] * len(tokenized_src[idx]))
            tokenized_labels.append(labels[labels[self.id] == patient].iloc[0, 1])
            idx += 1
        print("FINISHED TOKENIZATION. \n")
        return pd.DataFrame(tokenized_src), pd.DataFrame(tokenized_gender), pd.DataFrame(tokenized_ethni), pd.DataFrame(tokenized_ins), pd.DataFrame(tokenized_age), pd.DataFrame(tokenized_labels), pd.DataFrame(tokenized_labs), pd.DataFrame(tokenized_meds), meds_labels


    def tokenize(self):
        labs_list = []
        demo_list = []
        cond_list = []
        labels =  pd.read_csv('/datasets/MIMIC-IV/data/csv/'+'labels.csv')
        first = True
        labels = labels[0:5000]
        df_filter = pd.read_csv('/h/chloexq/los-prediction/pipeline/dynamic_item_dict_short_263.csv')
        id_filter = [int(i) for i in df_filter['itemid'].values]
        index_chart_only = df_filter[df_filter['type']=='CHART'].index.tolist()
        index_meds_only = df_filter[df_filter['type']=='MEDS'].index.tolist()
        print("STARTING READING FILES.")
        for hadm in tqdm.tqdm(labels.itertuples(), total = labels.shape[0]):
            labs = pd.read_csv('/datasets/MIMIC-IV/data/csv/' + str(hadm[1]) + '/dynamic.csv')
            labs = labs.loc[:, labs.iloc[0, :].isin(id_filter)].copy()
            
            demo = pd.read_csv('/datasets/MIMIC-IV/data/csv/' + str(hadm[1]) + '/demo.csv')
            cond = pd.read_csv('/datasets/MIMIC-IV/data/csv/' + str(hadm[1]) + '/static.csv')
            if first:
                condVocab_l = cond.iloc[0: , :].values.tolist()[0]
                first = False
            labs = labs.iloc[1: , :]
            cond = cond.iloc[1: , :]

            labs[self.id] = hadm[1]
            demo[self.id] = hadm[1]
            cond[self.id] = hadm[1]

            labs_list += labs.values.tolist()
            demo_list += demo.values.tolist()
            cond_list += cond.values.tolist()

        print("FINISHED READING FILES. \n")
        labs_list = pd.DataFrame(labs_list)
        demo_list = pd.DataFrame(demo_list)
        cond_list = pd.DataFrame(cond_list, columns=condVocab_l + [self.id])
        labs_list = labs_list.rename(columns={labs_list.columns.to_list()[-1]: self.id})
        demo_list = demo_list.rename(columns={demo_list.columns.to_list()[-1]: self.id})
        labs_list = pd.DataFrame(labs_list)
        demo_list = pd.DataFrame(demo_list)
        cond_list = pd.DataFrame(cond_list, columns=condVocab_l + [self.id])
        labs_list = labs_list.rename(columns={labs_list.columns.to_list()[-1]: self.id})
        demo_list = demo_list.rename(columns={demo_list.columns.to_list()[-1]: self.id})
        
        labs_list.replace(0, np.nan, inplace=True)
        '''    for col in labs_list.columns.to_list()[:-1]:
                if labs_list[col].nunique() < 2:
                    labs_list = labs_list.drop(columns=col)
        '''
        labs_codes = set()
        for col in labs_list.columns.to_list()[:-1]:
            labels_l = []
            if labs_list[col].nunique() > 1 :
                for i in range(len(pd.qcut(labs_list[col], 4, duplicates='drop', retbins=True)[1]) - 1):
                    labels_l.append(str(col) + "_" + str(i))
                labs_list[col] = pd.qcut(labs_list[col], 4, labels=labels_l, duplicates='drop')
                labs_codes.update(labels_l)
            elif labs_list[col].nunique() == 1 :
                labs_list.loc[labs_list[labs_list[col] > 0][col].index, col] = "dyn_" + str(col)
                labs_codes.add("dyn_" + str(col))

        ethVocab = {}
        insVocab = {}
        condVocab = {'token2idx': {}, 'idx2token': {0: 'PAD', 1: 'CLS', 2: 'SEP'}}
        with open('/datasets/MIMIC-IV/data/dict/ethVocab', 'rb') as fp:
            ethVocab_l = pickle.load(fp)
            for i in range(len(ethVocab_l)):
                ethVocab[ethVocab_l[i]] = i

        with open('/datasets/MIMIC-IV/data/dict/insVocab', 'rb') as fp:
            insVocab_l = pickle.load(fp)
            for i in range(len(insVocab_l)):
                insVocab[insVocab_l[i]] = i

        for v in condVocab_l:
            condVocab['idx2token'][max(condVocab['idx2token']) + 1] = v
        genderVocab = {'M': 0, 'F': 1}
        
        for new_code in labs_codes:
            condVocab['idx2token'][max(condVocab['idx2token']) + 1] = new_code

        condVocab['idx2token'][max(condVocab['idx2token']) + 1] = 'UNK'
        condVocab['token2idx'] = {v: k for k, v in condVocab['idx2token'].items()}
        cond_list = cond_list.sort_values(by=self.id)
        labs_list = labs_list.reset_index()
        labs_list = labs_list.sort_values(by=[self.id, 'index'])
        labs_list = labs_list.drop(columns=['index'])
        demo_list = demo_list.sort_values(by=self.id)

        labs_only_list = labs_list.loc[:, index_chart_only]
        meds_only_list = labs_list.loc[:, index_meds_only]

        labs_tokens = labs_only_list.values.astype(str).flatten()
        labs_tokens = np.unique(labs_tokens[labs_tokens!="nan"])
        labs_tokens = labs_tokens.tolist()

        meds_tokens = meds_only_list.values.astype(str).flatten()
        meds_tokens = np.unique(meds_tokens[meds_tokens!="nan"])
        meds_tokens = meds_tokens.tolist() + ["nan"]
        n_meds = len( meds_tokens)

        le = LabelEncoder()
        le.fit(meds_tokens)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        with open('./data/dict/condVocab.pkl', 'wb') as f:
            pickle.dump(condVocab, f)
        # with open('./data/dict/cond_vocab.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        with open('./data/dict/ethVocab.pkl', 'wb') as f:
            pickle.dump(ethVocab, f)
        
        with open('./data/dict/insVocab.pkl', 'wb') as f:
            pickle.dump(insVocab, f)
        
        with open('./data/dict/genderVocab.pkl', 'wb') as f:
            pickle.dump(genderVocab, f)

        tokenized_src, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_age, tokenized_labels, tokenized_labs, tokenized_meds, meds_labels = self.tokenize_dataset(
            labs_list, cond_list, demo_list, labels, condVocab, ethVocab, insVocab, genderVocab, labs_tokens, meds_tokens)
        
        for i in range(len(meds_labels)):
            meds_labels[i] = le.transform(meds_labels[i])

        print("FINAL COHORT STATISTICS: ")
        print(str(len(tokenized_labels[tokenized_labels[0] == 1])) + " Positive samples.")
        print(str(len(tokenized_labels[tokenized_labels[0] == 0])) + " Negative samples.\n")

        print(str(len(tokenized_gender[tokenized_gender[0] == 1])) + " Female samples.")
        print(str(len(tokenized_gender[tokenized_gender[0] == 0])) + " Male samples.\n")

        ethVocab_reversed = {v: k for k, v in ethVocab.items()}
        for i in range(len(ethVocab_reversed)):
            print(str(len(tokenized_ethni[tokenized_ethni[0] == i])) + " " + ethVocab_reversed[i] + " samples.")
        print("\n")

        insVocab_reversed = {v: k for k, v in insVocab.items()}
        for i in range(len(insVocab_reversed)):
            print(str(len(tokenized_ins[tokenized_ins[0] == i])) + " " + insVocab_reversed[i] + " samples.")
        
        # TODO: Fill NaN in dynamic data with PAD token value 
        # In other dfs such as age, gender etc., fill NaN with a new token to indicate missingness
        tokenized_src.fillna(0, inplace=True)
        tokenized_labs.fillna(0, inplace=True)
        tokenized_meds.fillna(0, inplace=True)

        tokenized_age.fillna(int(tokenized_age.max().max() + 1), inplace=True)
        # {'M': 0, 'F': 1s}
        tokenized_gender.fillna(len(genderVocab), inplace=True)
        # {0: 'BLACK/AFRICAN AMERICAN', 1: 'WHITE', 2: 'UNKNOWN', 3: 'BLACK/CAPE VERDEAN', 4: 'WHITE - BRAZILIAN', 5: 'BLACK/AFRICAN', 6: 'HISPANIC OR LATINO', 7: 'OTHER', 8: 'UNABLE TO OBTAIN', 9: 'HISPANIC/LATINO - SALVADORAN', 10: 'WHITE - OTHER EUROPEAN', 11: 'BLACK/CARIBBEAN ISLAND', 12: 'ASIAN', 13: 'HISPANIC/LATINO - DOMINICAN', 14: 'HISPANIC/LATINO - PUERTO RICAN', 15: 'WHITE - RUSSIAN', 16: 'ASIAN - KOREAN', 17: 'ASIAN - CHINESE', 18: 'PATIENT DECLINED TO ANSWER', 19: 'WHITE - EASTERN EUROPEAN', 20: 'ASIAN - ASIAN INDIAN', 21: 'ASIAN - SOUTH EAST ASIAN', 22: 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 23: 'AMERICAN INDIAN/ALASKA NATIVE', 24: 'HISPANIC/LATINO - CENTRAL AMERICAN', 25: 'PORTUGUESE', 26: 'HISPANIC/LATINO - GUATEMALAN', 27: 'HISPANIC/LATINO - HONDURAN', 28: 'HISPANIC/LATINO - CUBAN', 29: 'SOUTH AMERICAN', 30: 'MULTIPLE RACE/ETHNICITY', 31: 'HISPANIC/LATINO - COLUMBIAN', 32: 'HISPANIC/LATINO - MEXICAN'}
        tokenized_ethni.fillna(len(ethVocab), inplace=True)
        # {0: 'Medicare', 1: 'Other', 2: 'Medicaid'}  
        tokenized_ins.fillna(len(insVocab), inplace=True)
        return tokenized_src, tokenized_age, tokenized_gender, tokenized_ethni, tokenized_ins, tokenized_labels, tokenized_labs, tokenized_meds, meds_labels, n_meds
