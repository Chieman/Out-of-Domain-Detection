import pandas as pd
n_class_seen = 2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from model import BiLSTM
from sklearn import metrics
import argparse
import os
from tqdm import tqdm
from keras.utils import to_categorical
import copy
from transformers import AutoModel, AutoTokenizer, AutoConfig
import string




embedding_matrix = None
BATCH_SIZE = 50
HIDDEN_DIM = 128
CON_DIM = 32
NUM_LAYERS = 1
DO_NORM = True
ALPHA = 1.0
BETA = 1.0
OOD_LOSS = None
NORM_COEF = 0.1
CL_MODE = 1.0
LMCL = True
USE_BERT = True
SUP_CONT = True
CUDA = False
ADV = None
CONT_LOSS = None
model = BiLSTM(embedding_matrix, BATCH_SIZE, HIDDEN_DIM, CON_DIM, NUM_LAYERS, n_class_seen, DO_NORM, ALPHA, BETA,
                   OOD_LOSS, ADV, CONT_LOSS, NORM_COEF, CL_MODE, LMCL, use_bert=USE_BERT, sup_cont=SUP_CONT,
                   use_cuda=CUDA)
model.load_state_dict(copy.deepcopy(torch.load("/home/an/Documents/out-of-domain/Coding2/results/model_best.pt", map_location='cpu')))
model.eval()
valid_data = pd.read_csv("/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Valid_.csv")
valid_seen_text = list(valid_data['text'])
scores_result = []
with torch.no_grad():
    encode = model.bert_tokenizer(valid_seen_text, max_length=150, return_tensors='pt', padding=True, truncation=True)
    input_ids = encode["input_ids"]
    attention_mask = encode["attention_mask"]
    for i in range(len(encode["input_ids"])):
        input_ids_ = torch.unsqueeze(input_ids[i], dim=0)
        attention_mask_ = torch.unsqueeze(attention_mask[i], dim=0)
        seq_embed = model.bert_model(input_ids_, attention_mask_)[0]
        seq_embed = model.dropout(seq_embed)
        seq_embed = seq_embed.detach().float()
        scores, ht = model.rnn(seq_embed)
        ht = torch.cat((ht[0], ht[1]), dim=1)
        logits = model.fc(ht)
        probs = torch.softmax(logits, dim=1)
        scores_result.append(ht.detach().numpy())
print(np.array(scores_result).shape)
np.save("/home/an/Documents/out-of-domain/Coding2/results/valid_scores.npy", np.array(scores_result))
# for i in valid_seen_text:
#     print(u'')
#
#     seq_embed = model.bert_model(**model.bert_tokenizer(torch.tensor(i), max_length=150, return_tensors='pt', padding=True, truncation=True))[0]
#     seq_embed = model.dropout(seq_embed)
#     seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
#     scores, _ = model.rnn(seq_embed)
#     scores_result.append(scores)
# print(scores)
