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
  - 
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
```
2023-11-08 00:53:03.106712: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-11-08 00:53:03.106761: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-11-08 00:53:03.108388: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

## 天氣資料集
本教學使用[馬克斯普朗克生物地球化學研究所](https://www.bgc-jena.mpg.de/wetter/)記錄的[天氣時間序列資料集](https://www.bgc-jena.mpg.de/)。

此資料集包含了14 個不同特徵，例如氣溫、氣壓和濕度。自2003 年起，這些數據每10 分鐘就會被收集一次。為了提高效率，您將只使用2009 至2016 年之間收集的資料。資料集的這一部分由François Chollet 為他的[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)一書所準備。
```
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
13568290/13568290 [==============================] - 0s 0us/step
```
本教學僅處理**每小時預測**，因此先從10分鐘間隔到1小時對資料進行下採樣：
```
df = pd.read_csv(csv_path)
# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
```
讓我們看一下數據。下面是前幾行：
```
df.head()
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%861.png)
以下是一些特徵隨時間的演變：
```
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%862.png)
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%863.png)
### 檢查和清理
接下來，來看看資料集的統計數據：
```
df.describe().transpose()
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%864.png)
風速 值得注意的一件事是風速 (wv (m/s)) 的 min 值和最大值 (max. wv (m/s)) 列。這個 -9999 可能是錯的。

有一個單獨的風向列，因此速度應大於零 (>=0)。將其替換為零：
```
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()
```
```
0.0
```
### 特徵工程

在潛心建立模型之前，請務必了解資料並確保傳遞格式正確的資料。

風

資料的最後一列 wd (deg) 以度為單位給出了風向。角度不是很好的模型輸入：360° 和 0° 應該會彼此接近，並且平滑換行。如果不吹風，方向則無關緊要。

現在，風資料的分佈狀況如下：
```
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
```
```
Text(0, 0.5, 'Wind Velocity [m/s]')
```








###
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%866.png)
#### 時間
同樣，Date Time列非常有用，但不是以這種字串形式。首先將其轉換為秒：
```
timestamp_s = date_time.map(pd.Timestamp.timestamp)
```
與風向類似，以秒為單位的時間不是有用的模型輸入。作為天氣數據，它​​有清晰的每日和每年週期性。可以透過多種方式處理週期性。

您可以透過使用正弦和餘弦變換為清晰的“一天中的時間”和“一年中的時間”信號來獲得可用的信號：
```
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
```
```
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
```
```
Text(0.5, 1.0, 'Time of day signal')
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%867.png)

這使模型能夠存取最重要的頻率特徵。在這種情況下，您提前知道了哪些頻率很重要。

如果您沒有該資訊，則可以透過使用快速傅立葉變換提取特徵來確定哪些頻率重要。要檢驗假設，以下是溫度隨時間變化的tf.signal.rfft。請注意1/year和1/day附近頻率的明顯峰值：
```
fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%868.png)

###拆分數據
您將使用(70%, 20%, 10%)拆分出訓練集、驗證集和測試集。請注意，在拆分前資料沒有隨機打亂順序。這有兩個原因：

1.確保仍然可以將資料切入連續樣本的視窗。
2.確保訓練後在收集的數據上對模型進行評估，驗證/測試結果更加真實。
```
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
```
###歸一化數據
在訓練神經網路之前縮放特徵很重要。歸一化是進行此類縮放的常見方式：減去平均值，然後除以每個特徵的標準差。

平均值和標準偏差應僅使用訓練資料進行計算，從而使模型無法存取驗證集和測試集中的值。

有待商榷的是：模型在訓練時不應存取訓練集中的未來值，以及應該使用移動平均數來進行此類規範化。這不是本教學的重點，驗證集和測試集會確保我們獲得（某種程度上）可靠的指標。因此，為了簡單起見，本教學使用的是簡單平均數。
```
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
```
現在來看看這些特徵的分佈。部分特徵的尾部確實很長，但沒有類似-9999風速值的明顯錯誤。
```
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
```
```
<ipython-input-22-4ec9be458a7e>:5: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  _ = ax.set_xticklabels(df.keys(), rotation=90)
```

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%B3%87%E6%96%99%E9%9B%869.png)
##資料視窗化
本教程中的模型將基於來自資料連續樣本的視窗進行一組預測。

輸入視窗的主要特徵包括：

輸入和標籤視窗的寬度（時間步驟數）。
它們之間的時間偏移量。
用作輸入、標籤或兩者的特徵。
本教學建立了各種模型（包括線性、DNN、CNN 和RNN 模型），並將它們用於以下兩種情況：

單輸出和多重輸出預測。
單時間步驟和多時間步驟預測。
本部分重點在於實現資料視窗化，以便將其重複使用到上述所有模型。

根據任務和模型類型，您可能需要產生各種資料視窗。下面是一些範例：

例如，要在給定24 小時歷史記錄的情況下對未來24 小時作出一次預測，可以定義如下視窗：
```
```

```
```
# 參考資料
[時間序列預測]([https://www.cc.ntu.edu.tw/chinese/epaper/0052/20200320_5207.html](https://tensorflow.google.cn/tutorials/structured_data/time_series?hl=zh_cn))
