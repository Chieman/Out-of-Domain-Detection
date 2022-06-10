import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from metrics import fpr_at_x_tpr, roc_auc, roc_aupr
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Calculate mahalanobis or euclidean based confidence score for each class.
    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)
    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)

    features = features.reshape(num_samples, 1, num_features).repeat(num_classes,
                                                                     axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples,
                                                               axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result


def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])

    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)

    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + 1e-10

    print("estimated threshold:", threshold)
    return threshold


prob_train = np.load("/home/an/Documents/out-of-domain/Coding2/results/train_scores.npy")
print(prob_train.shape)
prob_train = np.squeeze(prob_train, axis=1)
prob_valid_ = np.load("/home/an/Documents/out-of-domain/Coding2/results/valid_scores.npy")
prob_valid_ = np.squeeze(prob_valid_, axis=1)
prob_valid = prob_valid_[:621, :]
prob_valid_ood = prob_valid_[621:, :]
# prob_train_ = []
# for i in prob_train:
#     prob_train_.append(np.array(i[0]))
# prob_train_ = np.array(prob_train_)
train_data = pd.read_csv("/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Train_.csv")
train_label = train_data['label']
valid_data = pd.read_csv("/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Valid_.csv")
valid_label = valid_data['label']
for i in range(len(valid_label)):
    if valid_label[i] == "OOD":
        valid_label[i] = 1
    else:
        valid_label[i] = 0

print(prob_train.shape, train_label.shape)
gda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, store_covariance=True)
gda.fit(prob_train, train_label)

# seen_m_dist = confidence(prob_valid, gda.means_, "mahalanobis", gda.covariance_).min(axis=1)
# unseen_m_dist = confidence(prob_valid_ood, gda.means_, "mahalanobis", gda.covariance_).min(axis=1)
all_m_dist = confidence(prob_valid_, gda.means_, "mahalanobis", gda.covariance_).min(axis=1)
is_ood = np.array([valid_label]).reshape((1991, ))
print(fpr_at_x_tpr(all_m_dist, is_ood.astype(int), 90, True))
seen_ood_score = []
unseen_ood_score = []
for i in range(len(all_m_dist)):
    if is_ood[i] == 0:
        seen_ood_score.append(all_m_dist[i])
    else:
        unseen_ood_score.append(all_m_dist[i])
# plt.hist(seen_ood_score, 100, alpha=0.5, label='In-Domain')
# plt.hist(unseen_ood_score, 100, alpha=0.5, label='Out-of-Domain')
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig("/home/an/Documents/out-of-domain/Coding2/results/hist.png")
y_true = is_ood
y_pred = []
for i in range(len(y_true)):
    if y_true[i] == 0:
        y_true[i] = "In-Domain"
    else:
        y_true[i] = "Out-of-Domain"
    if all_m_dist[i] < 15:
        y_pred.append("In-Domain")
    else:
        y_pred.append("Out-of-Domain")

y_pred = np.array(y_pred)
cf_matrix = confusion_matrix(y_true, y_pred)
classification_report(y_true, y_pred)





# valid_feats = np.squeeze(np.load("/home/an/Documents/out-of-domain/Coding2/results/valid_feats.npy"), axis=1)
# tsne = TSNE(n_components=2)
# result = tsne.fit_transform(valid_feats)
# sns.set(rc={'figure.figsize': (11.7, 8.27)})
# palette = sns.color_palette("bright", 2)
# sns.scatterplot(result[:, 0], result[:, 1], hue=is_ood, legend='full', palette=palette)
# plt.savefig("/home/an/Documents/out-of-domain/Coding2/results/val_feats.png")
