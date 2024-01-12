import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer

from config import Config


class WebDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = self.df_to_list(data_path)
        self.label = self.df_to_list(label_path)
        self.tokenizer = BertTokenizer.from_pretrained(Config.pretrain_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # print('--', data)
        data = self.tokenizer.encode_plus(data, padding='max_length', max_length=Config.max_length,
                                          truncation=True, return_tensors='pt')

        # 1 正常页面 0 被黑页面
        label = 1 if self.label[index] == 'n' else 0
        return data, label

    @staticmethod
    def df_to_list(path):
        df = pd.read_csv(path)
        return df.iloc[:, 0].tolist()

def collate_fn():
    pass
