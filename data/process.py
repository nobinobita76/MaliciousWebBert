import pandas as pd
import codecs
import chardet
import os

# 读取 Label 文件
columns = ['id', 'flag', 'filename', 'url']
origin_df = pd.read_csv('raw_data/label.txt', header=None, names=columns)

# n 正常页面 d 被黑页面
n_df = origin_df[origin_df['flag'] == 'n']
d_df = origin_df[origin_df['flag'] == 'd']

# 对正常样本下采样
under_n_df = n_df.sample(n=500)
df = pd.concat([under_n_df, d_df]).sample(frac=1)

file_list = os.listdir('processed_data/train_data')
for file in file_list:
    os.remove(os.path.join('processed_data/train_data', file))

# 统一编码为 utf-8
to_drop = []
for index, row in df.iterrows():
    if not os.path.exists(os.path.join('raw_data/train_data', row['filename'])):
        continue
    else:
        with codecs.open(os.path.join('raw_data/train_data', row['filename']), 'rb') as f:
            content = f.read()

        # 编码检测
        # {'GB2312': 344, 'utf-8': 287,
        # 'UTF-8-SIG': 17, 'ascii': 8, 'Windows-1254': 3, None: 1, 'ISO-8859-1': 1, 'MacRoman': 1}
        source_encoding = chardet.detect(content)['encoding']

        if not source_encoding:
            print(row['filename'], 'dropped for None encoding')
            to_drop.append(index)

        elif source_encoding == 'utf-8':
            with codecs.open(os.path.join('processed_data/train_data', row['filename']), 'wb') as f:
                f.write(content)

        else:
            # patch
            if source_encoding == 'GB2312':
                source_encoding = 'gb18030'
            if source_encoding == 'Windows-1254':
                source_encoding = 'utf-8'

            # print(source_encoding, filename, 'saving as utf-8')
            with codecs.open(os.path.join('processed_data/train_data', row['filename']), 'wb') as f:
                f.write(content.decode(source_encoding).encode('utf-8'))

df.drop(to_drop, inplace=True)

df.to_csv('processed_data/label.csv', header=False, index=False)


