import pywt
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

def filter_freq_discrete(x_input, t_ind, as_ind=1):
    x = x_input[:,t_ind,:].cpu().numpy()
    for i in range(len(x)):
        for j in range(len(t_ind)):
            if(as_ind==1):
                (cA, cD) = pywt.dwt(x[i,j], wavelet='haar', mode='symmetric', axis=-1)
                coeffs = (cA, cD)
                out = pywt.waverec(coeffs, wavelet='haar')
            else:
                (cA4, cD4, cD3, cD2, CD1) = pywt.wavedec(x[i,j], wavelet='haar', mode='symmetric', level=4, axis=-1)
                coeffs = (cA4, cD4, cD3, cD2, CD1)
                out = pywt.waverec(coeffs, wavelet='haar')
            #size_diff = len(out) - len(x_input[i,j])
            #print(len(out), len(x_input[i,j]))
            x_input[i,j] = torch.tensor(out)
    return x_input

def filter_freq(x_input, t_ind, as_ind=1):
    x = x_input[:,t_ind,:].cpu().numpy()
    ext_len = x.shape[2]//2
    for i in range(len(x)):
        for j in range(len(t_ind)):
            ext_data = pywt.pad(x[i,j], ext_len, mode='symmetric')
            cwtmatr, freqs = pywt.cwt(ext_data, scales=[as_ind], wavelet='morl')
            x_input[i,t_ind[j]] = torch.tensor(cwtmatr[-1, ext_len:-ext_len])
            #plt.plot(cwtmatr[-1, ext_len:-ext_len])
    return x_input