# 時間序列預測

南華大學 跨領域-人工智慧期末報告<br>
11024143 周舜鈞、11125038 劉皓群
****
本教學是使用TensorFlow 進行時間序列預測的簡介。它建構了幾種不同樣式的模型，包括卷積神經網路(CNN) 和循環神經網路(RNN)。

本教學包括兩個主要部分，每個部分包含若干小節：

預測單一時間步驟：
- 單一特徵。
  - 所有特徵。
- 預測多個時間步驟：
  - 單次：一次做出所有預測。
  - 自迴歸：一次做出一個預測，並將輸出饋送回模型。

## 安裝
```
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
```

## 天氣資料集
本教學使用馬克斯普朗克生物地球化學研究所記錄的[天氣時間序列資料集。

此資料集包含了14 個不同特徵，例如氣溫、氣壓和濕度。自2003 年起，這些數據每10 分鐘就會被收集一次。為了提高效率，您將只使用2009 至2016 年之間收集的資料。資料集的這一部分由François Chollet 為他的Deep Learning with Python一書所準備。
https://www.bgc-jena.mpg.de/wetter/
https://www.bgc-jena.mpg.de/
https://www.manning.com/books/deep-learning-with-python

###
![image]()


```
```

# 參考資料
[Google Colaboratory–適合Python初學者的雲端開發環境](https://www.cc.ntu.edu.tw/chinese/epaper/0052/20200320_5207.html)
