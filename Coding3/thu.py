# import csv
# import json
# csv_file_path = "/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Valid_.csv"
# data_dict = {}
# json_file_path = "/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Valid_.json"
# with open(csv_file_path, encoding='utf-8') as csv_file_handler:
#     csv_reader = csv.DictReader(csv_file_handler)
#     key = "oos_val"
#     data_dict[key] = []
#     for rows in csv_reader:
#         data_dict[key].append([rows['text'], rows['label']])
# with open(json_file_path, 'w') as json_file_handler:
#     # Step 4
#
#     json_file_handler.write(json.dumps(data_dict, indent=3, ensure_ascii=False))


from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from torch.nn import Linear, Dropout, Module
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from lib.datasets import datasets


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


class TransformerClassifier(Module):
    def __init__(self, transformer, hidden_dropout_prob, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.transformer = transformer
        self.dropout = Dropout(hidden_dropout_prob)
        self.classifier = Linear(self.transformer.config.hidden_size, self.num_labels)
        self.feats = None

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def forward(self, seq=None, att_idxs=None, inputs_embeds=None):
        bert_feats = self.transformer(input_ids=seq, attention_mask=att_idxs, inputs_embeds=inputs_embeds)
        self.feats = bert_feats[1]
        pooled_output = self.dropout(self.feats)
        return self.classifier(pooled_output), self.feats


def get_mean(texts, token_fn, transformer):
    result = []
    for text in texts:
        encoded = token_fn.batch_encode_plus(
            [text],
            max_length=150,
            return_tensors='pt',
            padding="max_length",
            truncation=True
        )
        padded_seq = encoded["input_ids"]
        att_idxs = encoded["attention_mask"]
        feats = transformer(padded_seq, att_idxs)[1]
        result.append(feats.detach().numpy())
    return np.array(result).mean(axis=0), np.array(result)


tokenizer, transformer = get_tokenizer_and_model("roberta")
pretrained_transformer_state_dict = torch.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/best_model.pt")
new_state_dict = {}
for key, value in pretrained_transformer_state_dict.items():
    if key != "classifier.weight" and key != "classifier.bias":
        new_state_dict[key[12:]] = value

transformer.load_state_dict(new_state_dict)
transformer.eval()

train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
                                                           # tokenizer=tokenizer,
                                                           dataset_name="Ours")
print(test_dataset[0])
# texts = [text[0] for text in train_dataset]
# with torch.no_grad():
#     _, train_feats = get_mean(texts, tokenizer, transformer)
# np.save("/home/an/Documents/out-of-domain/Coding3/debug/back_up/epoch2/train_feats.npy", train_feats)
# print(train_feats.shape)

train_feats = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/train_feats.npy")
train_feats = train_feats.reshape((len(train_feats), 768))
mean = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/mean.npy")
center_feats = []
for train_feat in train_feats:
    center_feats.append(train_feat - mean)
center_feats = np.array(center_feats)

pca = PCA(n_components=768).fit(center_feats)

in_texts = [text[0] for text in test_dataset]
print(in_texts)
encoded = tokenizer.batch_encode_plus(
            in_texts,
            max_length=150,
            return_tensors='pt',
            padding="max_length",
            truncation=True
        )
input_ids = encoded["input_ids"]
attention_mask = encoded['attention_mask']
result = []
with torch.no_grad():
    for i in range(len(in_texts)):
        feats = transformer(torch.unsqueeze(input_ids[i], 0), torch.unsqueeze(attention_mask[i], 0))[1]
        result.append(feats.detach().numpy())
        # r = feats - mean
        # r_components = pca.transform(r)
        # scores = np.power(r_components[:, 2:], 2) / \
        #          pca.explained_variance_[2:].reshape(1, -1)
        # ood_scores = torch.from_numpy(scores).sum(-1)
        # result.append(ood_scores[0])
np.save("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/val_feats.npy", np.array(result))
print(result)

