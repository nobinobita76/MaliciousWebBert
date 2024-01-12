import os.path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from bs4 import BeautifulSoup
import re


def get_info(file_path):
    # 提取网页文本信息
    html = BeautifulSoup(open(file_path, encoding='utf-8'), 'html.parser', from_encoding='utf-8')
    text = ''.join(html.stripped_strings)
    text = clean_text(text)
    return text[-512:]


def clean_text(string):
    # 去除非中文字符
    string = re.sub(r"[^\u4e00-\u9fa5]", "", string)
    # 去除回车和空格等空白字符
    string = re.sub(r"\s", "", string)
    return string


if __name__ == "__main__":
    columns = ['id', 'flag', 'filename', 'url']
    df = pd.read_csv('processed_data/label.csv', header=None, names=columns)

    # 提取文本信息
    df['content'] = ''
    for index, row in df.iterrows():
        df.loc[index, 'content'] = get_info(os.path.join('processed_data/train_data', row['filename']))

    # 删除空值
    df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    df.dropna(inplace=True)

    labels = df['flag']
    data = df['content']

    # 划分训练集和剩余的部分
    train_data, remaining_data, train_labels, remaining_labels = train_test_split(data, labels, test_size=0.2,
                                                                                  random_state=3407)
    # 划分剩余的部分为测试集和验证集
    test_data, val_data, test_labels, val_labels = train_test_split(remaining_data, remaining_labels, test_size=0.5,
                                                                    random_state=3407)

    # 打印划分后的数据集大小
    print("训练集大小:", len(train_data))
    print("测试集大小:", len(test_data))
    print("验证集大小:", len(val_data))

    train_data.to_csv('processed_data/train_data.csv', header=False, index=False)
    train_labels.to_csv('processed_data/train_label.csv', header=False, index=False)
    test_data.to_csv('processed_data/test_data.csv', header=False, index=False)
    test_labels.to_csv('processed_data/test_label.csv', header=False, index=False)
    val_data.to_csv('processed_data/val_data.csv', header=False, index=False)
    val_labels.to_csv('processed_data/val_label.csv', header=False, index=False)
