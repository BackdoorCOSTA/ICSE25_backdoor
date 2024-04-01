# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

    def forward(self, input_ids=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        return outputs



