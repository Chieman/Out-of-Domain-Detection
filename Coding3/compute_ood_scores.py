from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from lib.datasets import datasets
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.nn import Module


class AbstractMahalanobisScore(Module):
    def __init__(self, dim):
        super(AbstractMahalanobisScore, self).__init__()
        self.dim = dim
        self.register_buffer(
            'covariance_matrix',
            torch.eye(dim, dtype=torch.float)
        )

    def __call__(self, features):
        raise NotImplementedError

    def update(self, train_feats, train_labels):
        raise NotImplementedError

    def _check_scores(self, scores):
        if scores.dim() == 0:
            return scores.view(-1)
        return scores

    def update_inv_convmat(self, centered_feats):
        self.covariance_matrix.zero_()
        for feat in centered_feats:
            self.covariance_matrix += feat.view(-1, 1) @ feat.view(-1, 1).transpose(0, 1)
        self.covariance_matrix = self.covariance_matrix / centered_feats.shape[0]
        self.covariance_matrix = self.covariance_matrix.inverse()


class MarginalMahalanobisScore(AbstractMahalanobisScore):
    def __init__(self, dim):
        super(MarginalMahalanobisScore, self).__init__(dim)
        self.register_buffer(
            'mean',
            torch.zeros(dim, dtype=torch.float)
        )

    def __call__(self, features):
        r = features - self.mean
        r = r.unsqueeze(1)
        dist = r @ self.covariance_matrix @ r.transpose(1, 2)
        return self._check_scores(dist.squeeze())

    def center_feats(self, train_feats):
        centered_feats = torch.zeros_like(train_feats)
        for idx, feat in enumerate(train_feats):
            centered_feats[idx] = feat - self.mean
        return centered_feats

    def update(self, train_feats, train_labels):
        self.mean = train_feats.mean(dim=0)
        centered_feats = self.center_feats(train_feats)
        self.update_inv_convmat(centered_feats)


class MarginalMahalanobisPCAScore(MarginalMahalanobisScore):
    def __init__(self, dim, start_elem):
        super(MarginalMahalanobisPCAScore, self).__init__(dim)
        self.start_elem = start_elem
        self.pca = PCA(n_components=dim).fit(np.random.randn(dim, dim))

    def __call__(self, features):
        r = features - self.mean
        r_components = self.pca.transform(r.cpu().numpy())
        scores = np.power(r_components[:, self.start_elem:], 2) / \
                 self.pca.explained_variance_[self.start_elem:].reshape(1, -1)
        ood_scores = torch.from_numpy(scores).sum(-1)
        return self._check_scores(ood_scores)

    def update_pca(self, centered_feats):
        centered_feats = centered_feats.cpu().numpy()
        self.pca = PCA(n_components=self.dim).fit(centered_feats)

    def update(self, train_feats, train_labels):
        self.mean = train_feats.mean(dim=0)
        centered_feats = self.center_feats(train_feats)
        self.update_pca(centered_feats)


def load_best_model(transformer_type, path):
    if transformer_type == "PhoBert":
        model = AutoModel.from_pretrained("vinai/phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        new_state_dict = {}
        for (k, v) in torch.load(path).items():
            if "classifier" not in k:
                new_state_dict[k[12:]] = v
        model.load_state_dict(new_state_dict)
        model.eval()
    else:
        model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        old_state_dict = torch.load(path, map_location='cpu')["state_dict"]
        new_state_dict = {}
        for (k, v) in old_state_dict.items():
            if "classifier" not in k:
                if "score" not in k:
                    print(k[24:])
                    new_state_dict[k[24:]] = v
        print(new_state_dict.keys())
        model.load_state_dict(new_state_dict)
        model.eval()

    return model, tokenizer


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


def compute_ood_score(saved_path):
    train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
                                                           # tokenizer=tokenizer,
                                                           dataset_name="Ours")


if __name__ == "__main__":
    model, tokenizer = load_best_model("SimSCE", "/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/debug-epoch=03-val_loss=0.00.ckpt")
    train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
        # tokenizer=tokenizer,
        dataset_name="Ours")

    # texts = [text[0] for text in train_dataset]
    # with torch.no_grad():
    #     _, train_feats = get_mean(texts, tokenizer, model)
    # np.save("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/train_feats.npy", train_feats)
    # print(train_feats.shape)

    train_feats = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/train_feats.npy")
    train_feats = train_feats.reshape((len(train_feats), 768))
    mean = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/mean.npy")
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
            feats = model(torch.unsqueeze(input_ids[i], 0), torch.unsqueeze(attention_mask[i], 0))[1]
            result.append(feats.detach().numpy())
            # r = feats - mean
            # r_components = pca.transform(r)
            # scores = np.power(r_components[:, 2:], 2) / \
            #          pca.explained_variance_[2:].reshape(1, -1)
            # ood_scores = torch.from_numpy(scores).sum(-1)
            # result.append(ood_scores[0])
    np.save("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/val_feats.npy", np.array(result))
    print(result)

