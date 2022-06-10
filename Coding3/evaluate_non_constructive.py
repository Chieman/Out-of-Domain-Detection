from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from torch.nn import Linear, Dropout, Module
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from lib.datasets import datasets
import pandas as pd
from vncorenlp import VnCoreNLP


def get_tokenizer_and_model(model_name):
    if 'roberta' in model_name:
        model = AutoModel.from_pretrained("vinai/phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    elif 'distilbert' in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertWrapper(distil_bert_model=DistilBertModel.from_pretrained(model_name))
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    return tokenizer, model


tokenizer, transformer = get_tokenizer_and_model("roberta")
pretrained_transformer_state_dict = torch.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/best_model.pt")
new_state_dict = {}
for key, value in pretrained_transformer_state_dict.items():
    if key != "classifier.weight" and key != "classifier.bias":
        new_state_dict[key[12:]] = value

transformer.load_state_dict(new_state_dict)
transformer.eval()

# data = pd.read_csv("/home/an/Documents/out-of-domain/Coding/data/train/WS_Non-Constructive_Data.csv")
# text = [data['text'].values][0]
text = "Địt mẹ bọn chó đẻ cộng sản Việt Nam liếm bô cứt tàu cộng"
annotator = VnCoreNLP("/home/an/Documents/out-of-domain/Coding3/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg")
word_segmented_text = annotator.tokenize(text)
new_text = word_segmented_text[0][0]
for i in range(len(word_segmented_text)):
    if i == 0:
        for j in range(1, len(word_segmented_text[i])):
            new_text += " "
            new_text += word_segmented_text[i][j]
    else:
        for j in range(len(word_segmented_text[i])):
            new_text += " "
            new_text += word_segmented_text[i][j]
text = [new_text]
print(text, len(text))
train_feats = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/train_feats.npy")
train_feats = train_feats.reshape((len(train_feats), 768))
mean = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/mean.npy")
center_feats = []
for train_feat in train_feats:
    center_feats.append(train_feat - mean)
center_feats = np.array(center_feats)

pca = PCA(n_components=768).fit(center_feats)
encoded = tokenizer.batch_encode_plus(
            text,
            max_length=150,
            return_tensors='pt',
            padding="max_length",
            truncation=True
        )
input_ids = encoded["input_ids"]
attention_mask = encoded['attention_mask']
result = []
with torch.no_grad():
    for i in range(len(text)):
        feats = transformer(torch.unsqueeze(input_ids[i], 0), torch.unsqueeze(attention_mask[i], 0))[1]
        # result.append(feats.detach().numpy())
        r = feats - mean
        r_components = pca.transform(r)
        scores = np.power(r_components[:, 2:], 2) / \
                 pca.explained_variance_[2:].reshape(1, -1)
        ood_scores = torch.from_numpy(scores).sum(-1)
        result.append(ood_scores[0])
print(result)
# print(np.sum(np.array(result) < 2500))
# print(np.where(np.array(result) < 2500))
# print(text[13])
# for i in result:
#     if i < 2500:
#         print(i)
