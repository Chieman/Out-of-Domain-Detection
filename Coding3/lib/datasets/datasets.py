from collections import Counter
from typing import NoReturn

from nltk import word_tokenize
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from lib.data_utils import partial_class
from lib.datasets.loaders import My_Loader
SUBSETS = ("train", "val", "test")


class OODDataset(Dataset):
    """Defines a dataset for out-of-domain detection."""
    def __init__(self, loader, return_intent_labels=True, to_lower_case=True):
        """
        Create OODDataset given a data loader and a tokenization function.
        Args:
            loader: dataset loader
            tok_fn: tokenization function
            return_intent_labels: whether to return intent labels with instance (default: True)
            to_lower_case: whether to lowercase text data
        """
        super().__init__()
        self.loader = loader
        for attr in ["raw_texts", "raw_labels", "ood_labels"]:
            setattr(self, attr, getattr(self.loader, attr))
        self.n_ood = sum(self.ood_labels)
        self.n_indomain = len(self) - self.n_ood
        if to_lower_case:
            self.raw_texts = [t.lower() for t in self.raw_texts]
        # self.tokenized_texts = [tok_fn(t) for t in self.raw_texts]
        self.vectorized_texts = None
        self.return_intent_labels = return_intent_labels
        self.label_vocab, self.vectorized_labels, self.label_cnts = self.vectorize_labels()
        self.encoder = None

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, idx):
        if self.return_intent_labels:
            return self.raw_texts[idx], self.vectorized_labels[idx], self.ood_labels[idx]
        return self.raw_texts[idx], self.ood_labels[idx]

    def vectorize_labels(self):
        """
        Map raw labels onto their numerical representation.
        Returns:
            - label vocabulary, i.e mapping from labels to indexes
            - vectorized labels
            - label counts, i.e. number of instances in each class
        """
        label_counter = Counter(self.raw_labels)
        if 'OOD' in label_counter:
            label_counter.pop('OOD')
        unique_labels, label_cnts = zip(*sorted(label_counter.items()))
        unique_labels, label_cnts = list(unique_labels), list(label_cnts)
        label_vocab = {label: index for index, label in enumerate(unique_labels)}
        vectorized_labels = [label_vocab.get(label, -1) for label in self.raw_labels]
        return label_vocab, vectorized_labels, label_cnts

    # def vectorize_texts(self, encoder) -> NoReturn:
    #     """
    #     Map tokenized texts into respective numerical sequences.
    #     Args:
    #         encoder: mapping from tokens to integer values
    #     """
    #     self.encoder = encoder
    #     self.vectorized_texts = [self.encoder.encode(t) for t in self.tokenized_texts]


def get_transformer_splits(loader_cls, return_intent_labels=True):
    """
    Get train/dev/test split of the OOD dataset in the form suitable for transformer models.

    Args:
        loader_cls: class for loading OOD dataset from raw files.
        tokenizer: tokenizer from the `transformers` library
        return_intent_labels: whether to return intent labels with instances (default: True)
    Returns:
        list of dataset splits in order train/dev/test
    """
    datasets = []
    for subset in SUBSETS:
        dataset = OODDataset(loader_cls(subset=subset), return_intent_labels)
        dataset.raw_texts
        datasets.append(dataset)
        # dataset.vectorize_texts(tokenizer)
        # datasets.append(dataset)
    return datasets


def get_loader(dataset_name, **kwargs):
    if dataset_name == "Ours":
        loader = partial_class(My_Loader,
                               data_path="/home/an/Documents/out-of-domain/Coding/data/train/My_Data.json",
                               unsupervised=True)
    else:
        loader = partial_class(My_Loader,
                               data_path="/home/an/Documents/out-of-domain/Coding/data/train/My_Data.json",
                               unsupervised=True)
    return loader


def get_dataset_transformers(dataset_name, **kwargs):
    """
    Get OOD dataset splits.
    Args:
        tokenizer: tokenizer from `transformers` library related to a specific transformer
        dataset_name: name of the dataset
        **kwargs: additional arguments for a specific dataset
    Returns:
        train/dev/test split of the OOD dataset
    """
    loader = get_loader(dataset_name, **kwargs)
    return get_transformer_splits(loader)

