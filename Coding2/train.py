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


# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dataset", type=str, choices=["CLINC", "CLINC_OOD"], required=True,
                        help="The dataset to use, ATIS or SNIPS.")
    parser.add_argument("--proportion", type=int, required=True,
                        help="The proportion of seen classes, range from 0 to 100.")
    parser.add_argument("--seen_classes", type=str, nargs="+", default=None,
                        help="The specific seen classes.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both", "find_threshold"], default="both",
                        help="Specify running mode: only train, only test or both.")
    parser.add_argument("--setting", type=str, nargs="+", default=None,
                        help="The settings to detect ood samples, e.g. 'lof' or 'gda_lsqr")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="The directory contains model file (.h5), requried when test only.")
    parser.add_argument("--seen_classes_seed", type=int, default=None,
                        help="The random seed to randomly choose seen classes.")
    # default arguments
    parser.add_argument("--cuda", action="store_true",
                        help="Whether to use GPU or not.")
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="The gpu device to use.")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="The directory to store training models & logs.")
    parser.add_argument("--experiment_No", type=str, default="",
                        help="Manually setting of experiment number.")
    # model hyperparameters
    parser.add_argument("--embedding_file", type=str,
                        default="./glove_embeddings/glove.6B.300d.txt",
                        help="The embedding file to use.")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="The dimension of hidden state.")
    parser.add_argument("--contractive_dim", type=int, default=32,
                        help="The dimension of hidden state.")
    parser.add_argument("--embedding_dim", type=int, default=300,
                        help="The dimension of word embeddings.")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="The max sequence length. When set to None, it will be implied from data.")
    parser.add_argument("--max_num_words", type=int, default=10000,
                        help="The max number of words.")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="The layers number of lstm.")
    parser.add_argument("--do_normalization", type=bool, default=True,
                        help="whether to do normalization or not.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="relative weights of classified loss.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="relative weights of adversarial classified loss.")
    parser.add_argument("--unseen_proportion", type=int, default=100,
                        help="proportion of unseen class examples to add in, range from 0 to 100.")
    parser.add_argument("--mask_proportion", type=int, default=0,
                        help="proportion of seen class examples to mask, range from 0 to 100.")
    parser.add_argument("--ood_loss", action="store_true",
                        help="whether ood examples to backpropagate loss or not.")
    parser.add_argument("--adv", action="store_true",
                        help="whether to generate perturbation through adversarial attack.")
    parser.add_argument("--cont_loss", action="store_true",
                        help="whether to backpropagate contractive loss or not.")
    parser.add_argument("--norm_coef", type=float, default=0.1,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--n_plus_1", action="store_true",
                        help="treat out of distribution examples as the N+1 th class")
    parser.add_argument("--augment", action="store_true",
                        help="whether to use back translation to enhance the ood data")
    parser.add_argument("--cl_mode", type=int, default=1,
                        help="mode for computing contrastive loss")
    parser.add_argument("--lmcl", action="store_true",
                        help="whether to use LMCL loss")
    parser.add_argument("--cont_proportion", type=float, default=1.0,
                        help="coefficients of the normalized adversarial vectors")
    parser.add_argument("--dataset_proportion", type=float, default=100,
                        help="proportion for each in-domain data")
    parser.add_argument("--use_bert", action="store_true",
                        help="whether to use bert")
    parser.add_argument("--sup_cont", action="store_true",
                        help="whether to add supervised contrastive loss")
    # training hyperparameters
    parser.add_argument("--ind_pre_epoches", type=int, default=10,
                        help="Max epoches when in-domain pre-training.")
    parser.add_argument("--supcont_pre_epoches", type=int, default=100,
                        help="Max epoches when in-domain supervised contrastive pre-training.")
    parser.add_argument("--aug_pre_epoches", type=int, default=100,
                        help="Max epoches when adversarial contrastive training.")
    parser.add_argument("--finetune_epoches", type=int, default=20,
                        help="Max epoches when finetune model")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience when applying early stop.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Mini-batch size for train and validation")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="weight_decay")
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    args = parser.parse_args()
    return args


args = parse_args()
dataset = args.dataset
proportion = args.proportion
BETA = args.beta
ALPHA = args.alpha
DO_NORM = args.do_normalization
NUM_LAYERS = args.num_layers
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = args.batch_size
EMBEDDING_FILE = args.embedding_file
MAX_SEQ_LEN = args.max_seq_len
MAX_NUM_WORDS = args.max_num_words
EMBEDDING_DIM = args.embedding_dim
CON_DIM = args.contractive_dim
OOD_LOSS = args.ood_loss
CONT_LOSS = args.cont_loss
ADV = args.adv
NORM_COEF = args.norm_coef
LMCL = args.lmcl
CL_MODE = args.cl_mode
USE_BERT = args.use_bert
SUP_CONT = args.sup_cont
CUDA = args.cuda


#Tim index với các data seen (Pháp Luật + Chính Trị)
#Lấy dữ liệu raw từ các index đã được lấy
train_data = pd.read_csv("/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Train_.csv")
valid_data = pd.read_csv("/home/an/Documents/out-of-domain/Coding/data/train/Word_Seg_Train_.csv")
train_seen_text = list(train_data["text"])
valid_seen_text = list(valid_data['text'])
y_train_seen = list(train_data['label'])
y_valid_seen_ = list(valid_data['label'])
y_valid_seen = []
X_train_seen = ""
X_valid_seen = ""
for i in range(len(y_train_seen)):
    if y_train_seen[i] == "CT":
        y_train_seen[i] = 0
    else:
        y_train_seen[i] = 1
for i in range(len(y_valid_seen_)):
    if y_valid_seen_[i] == "CT":
        y_valid_seen.append(0)
    if y_valid_seen_[i] == "PL":
        y_valid_seen.append(1)
le = LabelEncoder()
le.fit(y_train_seen)
y_train_idx = le.transform(y_train_seen)
y_valid_idx = le.transform(y_valid_seen)

y_train_onehot = to_categorical(y_train_idx)
y_valid_onehot = to_categorical(y_valid_idx)

train_data_raw = (X_train_seen, y_train_onehot)
valid_data_raw = (X_valid_seen, y_valid_onehot)

output_dir = "/home/an/Documents/out-of-domain/Coding2/results"


class DataLoader(object):
    def __init__(self, data, batch_size, mode='train', use_bert=False, raw_text=None):
        self.use_bert = use_bert

        if self.use_bert:
            self.inp = list(raw_text)
        else:
            self.inp = data[0]
        self.tgt = data[1]
        self.batch_size = batch_size
        self.n_samples = len(self.inp)
        self.n_batches = self.n_samples // self.batch_size
        self.mode = mode
        self._shuffle_indices()

    def _shuffle_indices(self):
        if self.mode == 'test':
            self.indices = np.arange(self.n_samples)
        else:
            self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append((self.inp[_index],self.tgt[_index]))
            self.index += 1
            n += 1
        self.batch_index += 1
        seq, label = tuple(zip(*batch))
        if not self.use_bert:
            seq = torch.LongTensor(seq)
        if self.mode not in ['test', 'augment']:
            label = torch.FloatTensor(np.array(label))
        elif self.mode == 'augment':
            label = torch.LongTensor(label)

        return seq, label

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()


if args.mode in ["train", "both"]:
    embedding_matrix = None
    # filepath = os.path.join(output_dir, 'model_best.pt')
    model = BiLSTM(embedding_matrix, BATCH_SIZE, HIDDEN_DIM, CON_DIM, NUM_LAYERS, n_class_seen, DO_NORM, ALPHA, BETA,
                   OOD_LOSS, ADV, CONT_LOSS, NORM_COEF, CL_MODE, LMCL, use_bert=USE_BERT, sup_cont=SUP_CONT,
                   use_cuda=CUDA)
    model.load_state_dict(torch.load("/home/an/Documents/out-of-domain/Coding2/results/model_best.pt", map_location='cpu'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.cuda()

    # in-domain pre-training
    best_f1 = 0
    if args.sup_cont:
        for epoch in range(1, args.supcont_pre_epoches + 1):
            global_step = 0
            losses = []
            train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=train_seen_text)
            train_iterator = tqdm(train_loader, initial=global_step, desc="Iter (loss=X.XXX)")
            model.train()
            for j, (seq, label) in enumerate(train_iterator):
                if args.cuda:
                    if not USE_BERT:
                        seq = seq.cuda()
                    label = label.cuda()
                loss = model(seq, None, label, mode='ind_pre')
                train_iterator.set_description('Iter (sup_cont_loss=%5.3f)' % (loss.item()))
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                global_step += 1
                break
            print('Epoch: [{0}] :  Loss {loss:.4f}'.format(epoch, loss=sum(losses) / global_step))
            break
            # torch.save(model.state_dict(), filepath)

    # import pickle
    # model = torch.load(open("/home/an/Documents/out-of-domain/Coding2/results/model_best.pkl", 'rb'))
    # model.train()
    for epoch in range(1, args.ind_pre_epoches + 1):
        global_step = 0
        losses = []
        train_loader = DataLoader(train_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=train_seen_text)
        train_iterator = tqdm(train_loader, initial=global_step, desc="Iter (loss=X.XXX)")
        valid_loader = DataLoader(valid_data_raw, BATCH_SIZE, use_bert=USE_BERT, raw_text=valid_seen_text)
        model.train()
        for j, (seq, label) in enumerate(train_iterator):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            if epoch == 1:
                loss = model(seq, None, label, mode='finetune')
            else:
                loss = model(seq, None, label, sim=sim, mode='finetune')
            train_iterator.set_description('Iter (ce_loss=%5.3f)' % (loss.item()))
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            global_step += 1
            break
        print('Epoch: [{0}] :  Loss {loss:.4f}'.format(
            epoch, loss=sum(losses) / global_step))
        break
        model.eval()
        predict = []
        target = []
        if args.cuda:
            sim = torch.zeros((n_class_seen, HIDDEN_DIM * 2)).cuda()
        else:
            sim = torch.zeros((n_class_seen, HIDDEN_DIM * 2))
        for j, (seq, label) in enumerate(valid_loader):
            if args.cuda:
                if not USE_BERT:
                    seq = seq.cuda()
                label = label.cuda()
            output = model(seq, None, label, mode='validation')
            predict += output[0]
            target += output[1]
            sim += torch.mm(label.T, output[2])
        sim = sim / len(predict)
        n_sim = sim.norm(p=2, dim=1, keepdim=True)
        sim = (sim @ sim.t()) / (n_sim * n_sim.t()).clamp(min=1e-8)
        if args.cuda:
            sim = sim - 1e4 * torch.eye(n_class_seen).cuda()
        else:
            sim = sim - 1e4 * torch.eye(n_class_seen)
        sim = torch.softmax(sim, dim=1)
        f1 = metrics.f1_score(target, predict, average='macro')
        if f1 > best_f1:
            # torch.save(model, filepath)
            best_f1 = f1
        print('f1:{f1:.4f}'.format(f1=f1))
