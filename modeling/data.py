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


def data_collate_fn(dataset_samples_list, tokenizer):
    input = ["[CLS] " + i for i in dataset_samples_list]
    encoded = tokenizer.encode_batch(input)

    src_mask = torch.zeros(len(encoded[0].ids), len(encoded[0].ids))
    src_mask[1:, 0] = float("-inf")

    input = torch.tensor([i.ids for i in encoded])
    src_key_padding_mask = torch.tensor([i.special_tokens_mask for i in encoded])

    return {"input": input, "src_mask": src_mask, "src_key_padding_mask": src_key_padding_mask.bool()}
