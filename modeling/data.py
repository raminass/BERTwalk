import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BertWalkDataset(Dataset):
    def __init__(self, src, tokenizer):
        self.src = [i for i in src]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        return src


class TrainData(Dataset):
    def __init__(self, src, label, tokenizer):
        self.src = [i for i in src]
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.label[idx]


class TestData(Dataset):
    def __init__(self, src, label, tokenizer):
        self.src = [i for i in src]
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.label[idx]


def bert_walk_collate(dataset_samples_list, tokenizer):
    input = ["[CLS] " + i for i in dataset_samples_list]
    encoded = tokenizer.encode_batch(input)

    src_mask = torch.zeros(len(encoded[0].ids), len(encoded[0].ids))
    src_mask[1:, 0] = float("-inf")

    input = torch.tensor([i.ids for i in encoded])
    src_key_padding_mask = torch.tensor([i.special_tokens_mask for i in encoded])

    return {"input": input, "src_mask": src_mask, "src_key_padding_mask": src_key_padding_mask.bool()}


def classifier_collate(dataset_samples_list, tokenizer):
    input = ["[CLS] " + i[0] for i in dataset_samples_list]
    labels = [i[1] for i in dataset_samples_list]
    encoded = tokenizer.encode_batch(input)

    src_mask = torch.zeros(len(encoded[0].ids), len(encoded[0].ids))
    src_mask[1:, 0] = float("-inf")

    input = torch.tensor([i.ids for i in encoded])
    src_key_padding_mask = torch.tensor([i.special_tokens_mask for i in encoded])

    return {
        "input": input,
        "src_mask": src_mask,
        "src_key_padding_mask": src_key_padding_mask.bool(),
        "labels": torch.IntTensor(labels),
    }
