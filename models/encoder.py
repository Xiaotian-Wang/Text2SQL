import torch
from transformers import BertModel, BertTokenizer


class Encoder(torch.nn.Module):

    def __init__(self, pretrained_model_type):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_type)

    def forward(self, inputs):
