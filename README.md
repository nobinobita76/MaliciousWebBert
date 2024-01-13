# MaliciousWebBert
A Malicious (Chinese) Web Page Dectection Project Based on BERT, with GUI

Support uploading HTML files or inputting URLs to crawl page content

一个基于 BERT 的 (中文) 恶意网页检测工具，带有图形化界面

支持上传 HTML 文件或输入 URL 爬取页面内容

### Env
- Python
- PyTorch
- Transformers
- PyQT5

### Dataset
2017 中国网络安全技术对抗赛《恶意网页分析》(Incorporated)

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
