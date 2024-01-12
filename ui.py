import sys
import tempfile

import numpy as np
import requests
import torch
import codecs
import chardet
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit, QTextEdit, QFileDialog, \
    QTabWidget, QMainWindow, QHBoxLayout
from PyQt5.QtGui import QColor

from main import predict


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('恶意网页检测')
        self.setGeometry(100, 100, 400, 300)  # 设置窗口尺寸
        self.setFixedSize(400, 300)  # 设置主窗口固定大小
        # self.setStyleSheet("background-color: blue;")  # 设置背景色为白色

        # 创建一个主窗口部件和一个垂直布局
        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)

        # 创建一个标签页部件
        self.tab_widget = QTabWidget()

        # 创建 HTML 检测标签页
        self.html_tab = QWidget()
        self.html_layout = QVBoxLayout(self.html_tab)
        self.html_button = QPushButton("上传 HTML 文件")
        self.html_button.clicked.connect(self.upload_file)
        self.html_layout.addWidget(self.html_button)

        # 创建 URL 检测标签页
        self.url_tab = QWidget()
        self.url_layout = QVBoxLayout(self.url_tab)
        self.url_label = QLabel("请输入 URL:")
        self.url_input = QLineEdit()
        self.url_input.returnPressed.connect(self.process_url)
        self.url_layout.addWidget(self.url_label)
        self.url_layout.addWidget(self.url_input)

        # 将标签页添加到标签部件中
        self.tab_widget.addTab(self.html_tab, "HTML 检测")
        self.tab_widget.addTab(self.url_tab, "URL 检测")

        # 创建一个文本框部件来显示返回值
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        self.banner_label = QLabel("2020211866 王子俊 企业工程实践")
        # self.banner_label.setStyleSheet("font-size: 1px; font-weight: bold; color: #ff0000;")
        self.banner_label.setAlignment(QtCore.Qt.AlignCenter)  # 设置标签居中对齐

        # 创建水平布局并将标签添加到布局中
        self.banner_layout = QHBoxLayout()
        self.banner_layout.addWidget(self.banner_label)
        self.banner_layout.setContentsMargins(0, 10, 0, 10)  # 设置布局的边距

        self.main_layout.addWidget(self.tab_widget)
        self.main_layout.addWidget(self.result_text)
        self.main_layout.addWidget(self.banner_label)

        # 设置主窗口的中央部件
        self.setCentralWidget(self.main_widget)

    def process_url(self):
        url = self.url_input.text()
        result = process(mode='url', data=url)
        self.result_text.setText(result)

    def upload_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择文件')
        if file_path:
            result = process(mode='html', data=file_path)
            self.result_text.setText(result)


def process(mode, data):
    if mode == 'url':
        url = data

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/95.0.4638.69 Safari/537.36 '
        }

        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        html_content = response.text

    else:
        # HTML content
        with codecs.open(data, 'rb') as f:
            content = f.read()

        source_encoding = chardet.detect(content)['encoding']

        if not source_encoding:
            return '编码错误'

        elif source_encoding == 'utf-8':
            html_content = content

        else:
            # patch
            if source_encoding == 'GB2312':
                source_encoding = 'gb18030'
            if source_encoding == 'Windows-1254':
                source_encoding = 'utf-8'

            html_content = predict(content.decode(source_encoding).encode('utf-8'))

    # output = predict(html_content)
    # # return str(output.tolist())
    # return output

    flag, result, value = predict(html_content)

    if not flag:
        return '爬取到的内容过少'

    label_dict = {1: '正常', 0: '异常'}
    if value < 0.7:
        desc = '(低)\n此时检测结果置信度较低，网页情况尚不确定，请您仔细甄别。'
    elif 0.7 <= value < 0.9:
        desc = '(中)'
    else:
        desc = '(高)'

    res = f'{label_dict[result]}, 概率 {value:.6%} {desc}'

    return res


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
