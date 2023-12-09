import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import sys
import random
import pandas as pd
import pickle
import tqdm
import importlib
from pathlib import Path
from grad_reverse import AdversarialDiscriminator
from utils.filter_freq_util import filter_freq
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

#torch.manual_seed(42)
#torch.backends.cudnn.deterministic = True
class wave_act(nn.Module): 
    def __init__(self, scale=True, in_shape=[], training = True): 
        super(wave_act, self).__init__() 
        self.cos_val = 1.75 # 5, 1.75
        self.scale = scale
        if(self.scale):
            self.alpha = torch.nn.Parameter(torch.ones(in_shape))
            self.beta = torch.nn.Parameter(torch.zeros(in_shape))
            if training:
                self.alpha.requires_grad = True
                self.beta.requires_grad = True
            else:
                self.alpha.requires_grad = False
                self.beta.requires_grad = False
  
    def forward(self, x): 
        if self.scale:
            z = (x-self.beta)/self.alpha
        else:
            z = x
        return torch.cos(self.cos_val*z)*(torch.exp(-(z**2)/2))

def calc_activation_shape(dim, ksize, dilation=1, stride=1, padding=0):
    odim = dim + 2 * padding - dilation * (ksize - 1) - 1
    return int((odim / stride) + 1)

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.gender_embeddings = nn.Embedding(config.gender_vocab_size, config.hidden_size)
        self.ethnicity_embeddings = nn.Embedding(config.ethni_vocab_size, config.hidden_size)
        self.ins_embeddings = nn.Embedding(config.ins_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, labs_ids, age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None,
                posi_ids=None, age=True, if_include_meds=True):

        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)
        if if_include_meds:
            word_embed = self.word_embeddings(word_ids)
        else:
            word_embed = self.word_embeddings(labs_ids) 
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        gender_embed = self.gender_embeddings(gender_ids)
        ethnicity_embed = self.ethnicity_embeddings(ethni_ids)
        ins_embed = self.ins_embeddings(ins_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + gender_embed + ethnicity_embed + ins_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + gender_embed + ethnicity_embed + ins_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, conv_wavelet=False, training=True):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)
        self.conv_wavelet = conv_wavelet
        if(self.conv_wavelet):
            self.verbose = False
            hidden_size = config.hidden_size #288
            seq_len = 512 #2^9
            self.conv_shape = 'seq_DFT' # 'over_f' 'over_t' 'seq_DFT'
            if (self.conv_shape=='over_f'):
                dec_layer = []
                dec_layer.append(torch.nn.Conv1d(seq_len, seq_len//2, 1))
                dec_layer.append(torch.nn.Tanh())
                seq_len = seq_len//2
                self.verbose = False
                for i in range(4): #4, 2
                    dec_layer.append(torch.nn.Conv1d(seq_len, seq_len//4, 1)) #4, 16
                    if i>0:
                        dec_layer.append(wave_act(in_shape=[1,1,hidden_size], training=training))
                        #dec_layer.append(torch.nn.LayerNorm([seq_len//4, hidden_size]))
                        dec_layer.append(torch.nn.Tanh())
                    else:
                        dec_layer.append(torch.nn.Tanh())
                    seq_len = seq_len//4 #4, 16
                self.dec_layer = nn.ModuleList(dec_layer)
            elif (self.conv_shape=='over_t'):
                dec_layer = []
                """
                n_dilation = 5
                dilations = [2**i for i in range(n_dilation)]
                #cout_size = [hidden_size, hidden_size*2, hidden_size, hidden_size//2, hidden_size//4, hidden_size//8, hidden_size//16, hidden_size//32, hidden_size//64, hidden_size//128]
                cout_size = [hidden_size, hidden_size//2, hidden_size//4, hidden_size//8, hidden_size//16, hidden_size//32, hidden_size//64, hidden_size//128]
                for i in range(n_dilation): #4, 2
                    dec_layer.appe nd(torch.nn.Conv1d(cout_size[i], cout_size[i+1], 3, dilation=dilations[i], padding=2)) #4, 16
                    dec_layer.append(torch.nn.Tanh())
                self.dec_layer = nn.ModuleList(dec_layer)
                self.dec_layer_dense = (torch.nn.Linear(9*470, hidden_size)) # 18*498(kernel=3) 18*505(kernel=2)
                #"""
                self.verbose = False
                n_dilation = 5
                dilations = [2**i for i in range(n_dilation)]
                cout_size = [hidden_size]*(n_dilation+1)
                for i in range(n_dilation): #4, 2
                    dec_layer.append(torch.nn.Conv1d(cout_size[i], cout_size[i+1], 2, dilation=dilations[i], padding=1)) #4, 16
                    #dec_layer.append(nn.MaxPool1d(2, stride=2)) #return_indices=True
                    dec_layer.append(torch.nn.Tanh())
                    dec_layer.append(torch.nn.Dropout(0.2))
                dec_layer.append(nn.MaxPool1d(30, stride=30)) #20,19
                self.dec_layer = nn.ModuleList(dec_layer)
                self.dec_layer_dense = (torch.nn.Linear(288*16, hidden_size)) # 18*498(kernel=3) 18*505(kernel=2)
            else:
                # z = (x-a)/b
                dec_layer1 = []
                dec_layer2 = []
                final_dec_layer = []
                self.verbose = False
                n_dilation = 6 #5
                dilations = [2**i for i in range(n_dilation)]
                cout_size = [hidden_size]*(n_dilation+1)
                dim = seq_len//2
                for i in range(n_dilation): #4, 2
                    dec_layer1.append(torch.nn.Conv1d(cout_size[i], cout_size[i+1], 2, dilation=dilations[i], padding=1)) #4, 16
                    #dec_layer.append(nn.MaxPool1d(2, stride=2)) #return_indices=True
                    dim = calc_activation_shape(dim, 2, dilation=dilations[i], stride=1, padding=1)
                    #print(dim, hidden_size)
                    if i>1:
                        dec_layer1.append(wave_act(in_shape=[1,hidden_size,1], training=training))
                        #dec_layer1.append(torch.nn.LayerNorm([hidden_size, dim]))
                        dec_layer1.append(torch.nn.Tanh())
                    else:
                        dec_layer1.append(torch.nn.Tanh())
                    dec_layer1.append(torch.nn.Dropout(0.2))

                    dec_layer2.append(torch.nn.Conv1d(cout_size[i], cout_size[i+1], 2, dilation=dilations[i], padding=1)) #4, 16
                    #dec_layer.append(nn.MaxPool1d(2, stride=2)) #return_indices=True
                    if i>1:
                        dec_layer2.append(wave_act(in_shape=[1,hidden_size,1], training=training))
                        #dec_layer2.append(torch.nn.LayerNorm([hidden_size, dim]))
                        dec_layer2.append(torch.nn.Tanh())
                    else:
                        dec_layer2.append(torch.nn.Tanh())
                    dec_layer2.append(torch.nn.Dropout(0.2))
                self.dec_layer1 = nn.ModuleList(dec_layer1)
                self.dec_layer2 = nn.ModuleList(dec_layer2)
                final_dec_layer.append(torch.nn.Conv1d(cout_size[i], cout_size[i+1], 2, dilation=2**8, padding=1))
                final_dec_layer.append(torch.nn.Tanh())
                final_dec_layer.append(torch.nn.Dropout(0.2))
                final_dec_layer.append(nn.MaxPool1d(20, stride=19)) #20,19
                final_dec_layer.append(torch.nn.Linear(288*8, hidden_size))
                self.final_dec_layer = nn.ModuleList(final_dec_layer)


            ###causal version 
            #self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1)*dilation, dilation=dilation)

    def forward(self, input_ids, labs_ids, age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None,
                posi_ids=None, attention_mask=None, output_all_encoded_layers=True, 
                if_include_meds=False, wave_vis=False, filt_freq=False, t_ind=None, as_ind=None):
        conv_output= None
        wave_vec = []
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if gender_ids is None:
            gender_ids = torch.zeros_like(input_ids)
        if ethni_ids is None:
            ethni_ids = torch.zeros_like(input_ids)
        if ins_ids is None:
            ins_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, labs_ids, age_ids, gender_ids, ethni_ids, ins_ids, seg_ids, posi_ids, if_include_meds=if_include_meds)
        if filt_freq:
            embedding_output = filter_freq(embedding_output, t_ind, as_ind=as_ind)
        if wave_vis:
            wave_vec.append(embedding_output)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if self.conv_wavelet:
            if (self.conv_shape=='seq_DFT'):
                sequence_output = sequence_output.transpose(1,2)
                odd_seq = sequence_output[:,:,0::2]
                even_seq = sequence_output[:,:,1::2]
                for i, layer in enumerate(self.dec_layer1):
                    if i==0:
                        conv_output1 = layer(odd_seq)
                        if wave_vis:
                            wave_vec.append(conv_output1)
                    else:
                        conv_output1 = layer(conv_output1)
                    if self.verbose:
                        print('conv out shape: ', conv_output1.shape)
                    #if( wave_vis and (i==11)):
                    #    wave_vec.append(conv_output1)

                for i, layer in enumerate(self.dec_layer2):
                    if i==0:
                        conv_output2 = layer(even_seq)
                        if wave_vis:
                            wave_vec.append(conv_output2)
                    else:
                        conv_output2 = layer(conv_output2)
                    #if( wave_vis and (i==11)):
                    #    wave_vec.append(conv_output2)
                cat_out = torch.cat((conv_output1,conv_output2), 2)
                if self.verbose:
                    print('cat_out shape: ', cat_out.shape)
                for i, layer in enumerate(self.final_dec_layer):
                    if i==0:
                        out = layer(cat_out)
                        if wave_vis:
                            wave_vec.append(cat_out)
                            wave_vec.append(out)

                    elif i==(len(self.final_dec_layer)-1):
                        if self.verbose:
                            print('dec_out shape: ', out.shape)
                        out = out.reshape(out.shape[0], -1)
                        conv_output = layer(out)
                        conv_output = torch.nn.Tanh()(conv_output)
                    else: 
                        out = layer(out)
            else:
                if (self.conv_shape=='over_t'):
                    sequence_output = sequence_output.transpose(1,2)
                first_layer = True
                for layer in self.dec_layer:
                    if first_layer:
                        conv_output = layer(sequence_output)
                        first_layer = False
                    else:
                        conv_output = layer(conv_output)
                    if wave_vis:
                        wave_vec.append(conv_output)
                    if self.verbose:
                        print('conv out shape: ', conv_output.shape)
                if (self.conv_shape=='over_f'):
                    conv_output = conv_output.squeeze(1)
                else:
                    conv_output = conv_output.reshape(conv_output.shape[0], -1)
                    conv_output = self.dec_layer_dense(conv_output)
                    conv_output = torch.nn.Tanh()(conv_output)
        if(wave_vis):
            return conv_output, wave_vec
        return encoded_layers, pooled_output, conv_output


class BertForEHRPrediction(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels, conv_wavelet=True):
        super(BertForEHRPrediction, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, conv_wavelet = conv_wavelet)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.med_grad_reverse_classifier = AdversarialDiscriminator(config.hidden_size, n_cls=config.number_meds, reverse_grad=True)
        self.apply(self.init_bert_weights)
        self.conv_wavelet = conv_wavelet

    def forward(self, input_ids, labs_ids,age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, if_dab=False):
        _, pooled_output, cn_out = self.bert(input_ids, labs_ids, age_ids, gender_ids, ethni_ids, ins_ids, seg_ids, 
                                              posi_ids,attention_mask,output_all_encoded_layers=False, if_include_meds=True)
        
        pooled_output = self.dropout(pooled_output)
        if(self.conv_wavelet):
            logits = self.classifier(cn_out)
        else:
            logits = self.classifier(pooled_output)
        if if_dab:
            _, pooled_output_no_meds, cn_out_no_meds = self.bert(input_ids, labs_ids, age_ids, gender_ids, ethni_ids, ins_ids, seg_ids, posi_ids, attention_mask,output_all_encoded_layers=False, if_include_meds=False)
            
            pooled_output_no_meds = self.dropout(pooled_output_no_meds)
            if(self.conv_wavelet):
                logits_meds = self.med_grad_reverse_classifier(cn_out_no_meds)
            else:
                logits_meds = self.med_grad_reverse_classifier(pooled_output_no_meds)
        else:
            logits_meds = logits

        return logits, logits_meds
    
    def wave_test(self, input_ids, labs_ids,age_ids=None, gender_ids=None, ethni_ids=None, ins_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, if_dab=False, filt_freq=False, t_ind=None, as_ind=None):
        cn_out, wave_vec = self.bert(input_ids, labs_ids, age_ids, gender_ids, ethni_ids, ins_ids, seg_ids, 
                             posi_ids,attention_mask,output_all_encoded_layers=False, 
                             if_include_meds=True, wave_vis=True, filt_freq=filt_freq, t_ind=t_ind, as_ind=as_ind)
        logits = self.classifier(cn_out)

        return logits, wave_vec
        

class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.gender_vocab_size = config.get('gender_vocab_size')
        self.ethni_vocab_size = config.get('ethni_vocab_size')
        self.ins_vocab_size = config.get('ins_vocab_size')
        self.number_output = config.get('number_output')
        self.number_meds = config.get('number_meds')

class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')


class DataLoader(Dataset):
    def __init__(self, dataframe, max_len, code='code', age='age', labels='labels'):
        self.max_len = max_len
        self.code = dataframe[code]
        self.age = dataframe[age]
        self.labels = dataframe[labels]
        self.gender = dataframe["gender"]
        self.ethni = dataframe["ethni"]
        self.ins = dataframe["ins"]
        self.labs = dataframe["labs"]
        self.meds = dataframe["meds"]
        self.meds_labels = dataframe["meds_labels"]
        self.n_meds = dataframe["n_meds"]

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index]
        code = self.code[index]
        label = self.labels[index]
        gender = self.gender[index]
        ethni = self.ethni[index]
        ins = self.ins[index]
        labs = self.labs[index]
        meds = self.meds[index]
        meds_labels = nn.functional.one_hot(torch.tensor(self.meds_labels[index]), num_classes=self.n_meds)
        meds_labels = meds_labels.sum(0)

        # mask 0:len(code) to 1, padding to be 0
        # TODO: Update padding
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len)
        gender = seq_padding(gender, self.max_len)
        ethni = seq_padding(ethni, self.max_len)
        ins = seq_padding(ins, self.max_len)

        # get position code and segment code
        code = seq_padding(code, self.max_len)
        position = position_idx(code)
        segment = index_seg(code)

        return torch.LongTensor(code), torch.LongTensor(age), torch.LongTensor(gender), torch.LongTensor(
            ethni), torch.LongTensor(ins), \
               torch.LongTensor(segment), torch.LongTensor(position), \
               torch.FloatTensor(mask), torch.FloatTensor(label), \
               torch.LongTensor(labs), torch.LongTensor(meds), torch.LongTensor(meds_labels)

    def __len__(self):
        return len(self.code)

SEP = 2
PAD = 0

def seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = PAD

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[i])
        else:
            seq.append(symbol)
    return seq


def position_idx(tokens, symbol=SEP):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def index_seg(tokens, symbol=SEP):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)