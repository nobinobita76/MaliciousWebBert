# MaliciousWebBert
A Malicious (Chinese) Web Page Dectection Project Based on BERT, with GUI

Support uploading HTML files or inputting URLs to crawl page content

### Env
- Python
- PyTorch
- Transformers
- PyQT5

### Dataset
2017 China Cybersecurity Technology Competition 'Malicious Webpage Analysis'(Incorporated)

### BERT Pretrain Model
[Chinese-BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)

### How to Run
1. Download Pretain Model (for pytorch) to path model_hub/chinese_wwm_ext_pytorch
2. data/process.py and data/process_2.py to preprocess
3. Train and Test
   
   ```
   python main.py train
   python main.py test
   ```
5. Predict with GUI
   
   ```
   python ui.py
   ```

## 简介
一个基于 BERT 的 (中文) 恶意网页检测工具，带有图形化界面

支持上传 HTML 文件或输入 URL 爬取页面内容

### Env
- Python
- PyTorch
- Transformers
- PyQT5

### Dataset
2017 中国网络安全技术对抗赛《恶意网页分析》(已包含)

### 运行
1. 将预训练模型（PyTorch版本）下载到 model_hub/chinese_wwm_ext_pytorch 路径

2. 使用 data/process.py 和 data/process_2.py 进行数据预处理

3. 训练和测试
   ```
   python main.py train
   python main.py test
   ```

4. 使用图形界面进行预测
   ```
   python ui.py
   ```
