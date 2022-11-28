# create by luohao in 2022/11/28
# 拉取请求可让你在 GitHub 上向他人告知你已经推送到存储库中分支的更改。 在拉取请求打开后，
# 你可以与协作者讨论并审查潜在更改，
# 在更改合并到基本分支之前添加跟进提交。

# Pyside6是Qt6对应的官方python库，官方有较全面的使用文档：https://doc.qt.io/qtforpython/contents.html。
import os


from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog

# PySide6-uic demo.ui -o ui_demo.py 
# 编译ui文件成py文件
# 也有ui编译成cpp文件的命令行

from ui_demo import Ui_MainWindow

import argparse
import functools

import numpy as np
import torch

from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments




class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()  # UI类的实例化（）
        self.ui.setupUi(self)
        self.bind()
        self.setWindowTitle("语言相似度匹配")
        self.setWindowIcon(QPixmap("D:\photo\98072968_p1.jpg"))
        global str1
        global str2
        self.player = QMediaPlayer()
        self.audioOutput = QAudioOutput()  # 不能实例化为临时变量，否则被自动回收导致无法播放
        self.player.setAudioOutput(self.audioOutput)
        # Qt6中`QMediaPlayer.setVolume`已被移除，使用`QAudioOutput.setVolume`替代
        self.audioOutput.setVolume(1)

    def bind(self):
        # self.ui.___ACTION___.triggered.connect(___FUNCTION___)
        # self.ui.___BUTTON___.clicked.connect(___FUNCTION___)
        # self.ui.___COMBO_BOX___.currentIndexChanged.connect(___FUNCTION___)
        # self.ui.___SPIN_BOX___.valueChanged.connect(___FUNCTION___)
        # 自定义信号.属性名.connect(___FUNCTION___)

        self.ui.aupushbutton.clicked.connect(self.handle_click1)
        self.ui.au1pushbutton.clicked.connect(self.handle_click2)
        self.ui.startpushbutton.clicked.connect(self.handle_click3)

    def handle_click1(self):
        para1, para2 = QFileDialog.getOpenFileName(self, caption='选择音频文件', dir=os.getcwd(), filter="选择音频文件(*.wav)")
        print(para1)
        global str1
        str1= str(para1)


        st = str1 + "\n"
        self.ui.label.setText(st)

        self.player.setSource(QUrl.fromLocalFile(str1))
        self.player.play()


    def handle_click2(self):
        para1, para2 = QFileDialog.getOpenFileName(self, caption='选择音频文件', dir=os.getcwd(), filter="选择音频文件(*.wav)")
        print(para1)
        global str2
        str2 = str(para1)

        st = self.ui.label.text()
        st = st+str2+"\n"
        self.ui.label.setText(st)


        self.player.setSource(QUrl.fromLocalFile(str2))
        self.player.play()

    def handle_click3(self):
        global str1,str2
        parser = argparse.ArgumentParser(description=__doc__)
        add_arg = functools.partial(add_arguments, argparser=parser)
        add_arg('audio_path1', str, str1, '预测第一个音频')
        add_arg('audio_path2', str, str2, '预测第二个音频')
        add_arg('threshold', float, 0.33, '判断是否为同一个人的阈值')
        add_arg('input_shape', str, '(1, 257, 100)', '数据输入的形状')
        add_arg('model_path', str, 'resnet34.pth', '预测模型的路径')
        args = parser.parse_args()
        str3 = self.ui.label.text()+"-----------  Configuration Arguments -----------"+"\n"
        for arg, value in sorted(vars(args).items()):
            str3 += arg + ": " + str(value) + "\n"
        str3 += "------------------------------------------------"+"\n"
        # print_arguments(args)
        self.ui.label.setText(str3)

        device = torch.device("cpu")

        # 加载模型
        model = torch.jit.load(args.model_path, map_location='cpu')
        model.to(device)
        model.eval()

        # 预测音频
        def infer(audio_path):
            input_shape = eval(args.input_shape)
            data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
            data = data[np.newaxis, :]
            data = torch.tensor(data, dtype=torch.float32, device=device)
            # 执行预测
            feature = model(data)
            return feature.data.cpu().numpy()
        # 要预测的两个人的音频文件
        feature1 = infer(args.audio_path1)[0]
        feature2 = infer(args.audio_path2)[0]
        # 对角余弦值
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        if dist > args.threshold:
            str1 = str("%s 和 %s 为同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
            st = self.ui.label.text()
            st += str1
            self.ui.label.setText(st)

        else:
            str2 = str("%s 和 %s 不是同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))

            st = self.ui.label.text()
            st += str2
            self.ui.label.setText(st)








if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()





