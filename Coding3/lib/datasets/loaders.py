import json
from pathlib import Path

from pandas import read_csv
import pandas as pd


class Loader:
    """
    Abstract data loader.
    """
    def __init__(self, *, subset):
        assert subset in ["train", "val", "test"]


class My_Loader(Loader):
    """
    Data loader for CLINC150 dataset.
    """
    def __init__(self, data_path, subset, unsupervised=True):
        """
        Load CLINC150 subset.
        Args:
            data_path: path to the data file
            subset: subset to load (train/val/test)
            unsupervised: whether to use unsupervised version of the dataset (default: True)
        """
        super().__init__(subset=subset)
        with open(data_path) as f:
            data = json.load(f)
        # no ood samples used during training
        if subset == "train":
            if unsupervised:
                self.data_pairs = data["train"], list()
            else:
                raise NotImplementedError("Only unsupervised mode supported")
        elif subset == "val":
            self.data_pairs = data["val"], data["oos_val"]
        else:
            self.data_pairs = data["test"], data["oos_test"]
        self.n_indomain, self.n_ood = len(self.data_pairs[0]), len(self.data_pairs[1])
        self.ood_labels = [0] * self.n_indomain + [1] * self.n_ood
        self.data_pairs = self.data_pairs[0] + self.data_pairs[1]
        self.raw_texts, self.raw_labels = zip(*self.data_pairs)