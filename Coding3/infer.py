import torch
import numpy as np
from lib.utils import compute_ood_metrics
from lib.datasets import datasets
from lib.metrics import fpr_at_x_tpr, roc_auc, roc_aupr
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
                                                           # tokenizer=tokenizer,
                                                           dataset_name="Ours")
ood_scores = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/ood_scores.npy")
print(ood_scores.shape)

is_ood = [is_ood[2] for is_ood in test_dataset]


fpr95_in = fpr_at_x_tpr(ood_scores, is_ood, 95)
print(fpr95_in)

# train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
#                                                            # tokenizer=tokenizer,
#                                                            dataset_name="Ours")
# train_feats = np.load("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/val_feats.npy")
# is_ood = [i[2] for i in test_dataset]
# for i in range(len(is_ood)):
#     if is_ood[i] == 0:
#         is_ood[i] = "In-Domain"
#     else:
#         is_ood[i] = "Out-Of-Domain"
# np.save("/home/an/Documents/out-of-domain/Coding3/debug/back_up/PhoBert/best/y_True", is_ood)
# seen_ood_score = []
# unseen_ood_score = []
# for i in range(len(ood_scores)):
#     if is_ood[i] == "In-Domain":
#         seen_ood_score.append(ood_scores[i])
#     else:
#         unseen_ood_score.append(ood_scores[i])

# tsne = TSNE(n_components=2)
# result = tsne.fit_transform(train_feats.reshape((len(train_feats), 768)))
# sns.set(rc={'figure.figsize': (11.7, 8.27)})
# palette = sns.color_palette("bright", 2)
# sns.scatterplot(result[:, 0], result[:, 1], hue=is_ood, legend='full', palette=palette)
# plt.savefig("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/val_feats.png")
# plt.hist(seen_ood_score, 100, alpha=0.5, label='In-Domain')
# plt.hist(unseen_ood_score, 100, alpha=0.5, label='Out-of-Domain')
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig("/home/an/Documents/out-of-domain/Coding3/debug/back_up/SimCSE/hist.png")