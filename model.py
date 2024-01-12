import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from config import Config

class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.pretrain_path)
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooler_output = self.bert(input_ids, attention_mask, return_dict=False)
        dropout_output = self.dropout(pooler_output)
        fc_output = self.fc(dropout_output)
        logits = self.relu(fc_output)
        probabilities = nn.functional.softmax(logits, dim=1)

        return probabilities
