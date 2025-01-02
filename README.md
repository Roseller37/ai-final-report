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


## 資料視窗化
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

1.例如，要在給定24 小時歷史記錄的情況下對未來24 小時作出一次預測，可以定義如下視窗：
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E8%A6%96%E7%AA%97%E5%8C%961.png)

1.給定6 小時的歷史記錄，對未來1 小時作出一次預測的模型將需要類似下面的視窗：

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E8%A6%96%E7%AA%97%E5%8C%962.png)
本部分的剩餘內容會定義WindowGenerator類別。此類可以：

1.處理如上圖所示的索引和偏移量。

2.將特徵視窗拆分為(features, labels)對。

3.繪製結果視窗的內容。

4.使用tf.data.Dataset從訓練、評估和測試資料高效產生這些視窗的批次。
```
```
![image]()


### 1.索引和偏移量
首先創建WindowGenerator類別。__init__方法包含輸入和標籤索引的所有必要邏輯。

它還將訓練、評估和測試DataFrame 作為輸出。這些稍後將被轉換為視窗的tf.data.Dataset。
```
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
```
以下是建立本部分開頭圖表所示的兩個視窗的程式碼：
```
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['T (degC)'])
w1
```
```
Total window size: 48
Input indices: [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Label indices: [47]
Label column name(s): ['T (degC)']
```
```
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['T (degC)'])
w2
```
```
Total window size: 7
Input indices: [0 1 2 3 4 5]
Label indices: [6]
Label column name(s): ['T (degC)']
```
### 2. 拆分
給定一個連續輸入的列表，split_window方法會將它們轉換為輸入視窗和標籤視窗。

您之前定義的範例w2將按以下方式拆分：
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E8%A6%96%E7%AA%97%E5%8C%963.png)

此圖不顯示資料的features軸，但此split_window函數也會處理label_columns，因此可以用於單輸出和多輸出樣本。
```
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window
```
試試以下程式碼：
```
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
```
```
All shapes are: (batch, time, features)
Window shape: (3, 7, 19)
Inputs shape: (3, 6, 19)
Labels shape: (3, 1, 1)
```
通常，TensorFlow 中的資料會被打包到陣列中，其中最外層索引是交叉樣本（「批次」維度）。中間索引是「時間」和「空間」（寬度、高度）維度。最內層索引是特徵。

上面的程式碼使用了三個7 時間步驟視窗的批次，每個時間步驟有19 個特徵。它將其拆分成一個6 時間步驟的批次、19 個特徵輸入和一個1 時間步驟1 特徵的標籤。此標籤僅有一個特徵，因為WindowGenerator已使用label_columns=['T (degC)']進行了初始化。最初，本教學將建立預測單一輸出標籤的模型。

### 3. 繪圖
下面是一個繪圖方法，可以對拆分視窗進行簡單視覺化：
```
w2.example = example_inputs, example_labels
```

```
def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot
```
此繪圖根據項目引用的時間來對齊輸入、標籤和（稍後的）預測：
```
w2.plot()
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E8%A6%96%E7%AA%97%E5%8C%964.png)

您可以繪製其他列，但是樣本視窗w2設定僅包含T (degC)列的標籤。
```
w2.plot(plot_col='p (mbar)')
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E8%A6%96%E7%AA%97%E5%8C%965.png)

### 4. 創建tf.data.Dataset
最後，此make_dataset方法將取得時間序列DataFrame 並使用tf.keras.utils.timeseries_dataset_from_array函數將其轉換為(input_window, label_window)對的tf.data.Dataset。
```
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset
```
WindowGenerator物件包含訓練、驗證和測試資料。

使用您之前定義的make_dataset方法新增屬性以作為tf.data.Dataset存取它們。此外，新增一個標準樣本批次以便於存取和繪圖：
```
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
```
現在，WindowGenerator對象允許您存取tf.data.Dataset對象，因此您可以輕鬆迭代資料。

Dataset.element_spec屬性會告訴您資料集元素的結構、資料類型和形狀。
```
# Each element is an (inputs, label) pair.
w2.train.element_spec
```
```
(TensorSpec(shape=(None, 6, 19), dtype=tf.float32, name=None),
 TensorSpec(shape=(None, 1, 1), dtype=tf.float32, name=None))
```
在Dataset上進行迭代會產生具體批次：
```
for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
```
```
Inputs shape (batch, time, features): (32, 6, 19)
Labels shape (batch, time, features): (32, 1, 1)
```
## 單步模型
基於此類資料能夠建立的最簡單模型，能夠僅根據當前條件預測單一特徵的值，即未來的一個時間步驟（1 小時）。

因此，從建立模型開始，預測未來1 小時的T (degC)值。

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E5%AE%A4%E7%AA%97%E5%8C%966.png)

配置WindowGenerator物件以產生下列單步(input, label)對：
```
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])
single_step_window
```
```
Total window size: 2
Input indices: [0]
Label indices: [1]
Label column name(s): ['T (degC)']
```
window會根據訓練、驗證和測試集創建tf.data.Datasets，讓您可以輕鬆迭代資料批次。
```
for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
```

```
Inputs shape (batch, time, features): (32, 1, 19)
Labels shape (batch, time, features): (32, 1, 1)
```
### 基線
在建立可訓練模型之前，最好將效能基準作為與以後更複雜的模型進行比較的點。

第一個任務是在給定所有特徵的當前值的情況下，預測未來1 小時的溫度。目前值包括當前溫度。

因此，從僅返回當前溫度作為預測值的模型開始，預測「無變化」。這是一個合理的基線，因為溫度變化緩慢。當然，如果您對更遠的未來進行預測，此基線的效果就不那麼好了。

將輸入傳送到輸出
```
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]
```
實例化並評估此模型：
```
baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
```

```
```
```
```
上面的程式碼列印了一些效能指標，但這些指標並沒有使您對模型的運作有所了解。

WindowGenerator有一種繪製方法，但只有一個樣本，繪圖不是很有趣。

因此，創建一個更寬的WindowGenerator來一次產生包含24 小時連續輸入和標籤的視窗。新的wide_window變數不會改變模型的運算方式。模型仍會根據單一輸入時間步驟對未來1 小時進行預測。這裡time軸的作用類似batch軸：每個預測都是獨立進行的，時間步驟之間沒有交互作用：
```
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)'])

wide_window
```
```
439/439 [━━━━━━━━━━━━━━━━━━━━] 2s 4ms/step - loss: 0.0132 - mean_absolute_error: 0.0795
```
上面的程式碼列印了一些效能指標，但這些指標並沒有使您對模型的運作有所了解。

WindowGenerator有一種繪製方法，但只有一個樣本，繪圖不是很有趣。

因此，創建一個更寬的WindowGenerator來一次產生包含24 小時連續輸入和標籤的視窗。新的wide_window變數不會改變模型的運算方式。模型仍會根據單一輸入時間步驟對未來1 小時進行預測。這裡time軸的作用類似batch軸：每個預測都是獨立進行的，時間步驟之間沒有交互作用：
```
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)'])

wide_window
```
```
Total window size: 25
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Label indices: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
Label column name(s): ['T (degC)']
```
此擴充視窗可以直接傳遞到相同的baseline模型，而無需修改任何程式碼。能做到這一點是因為輸入和標籤具有相同數量的時間步驟，並且基線只是將輸入轉發至輸出：
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E5%AE%A4%E7%AA%97%E5%8C%967.png)

```
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
```

```
```
![image]()

# 參考資料
[時間序列預測]([https://www.cc.ntu.edu.tw/chinese/epaper/0052/20200320_5207.html](https://tensorflow.google.cn/tutorials/structured_data/time_series?hl=zh_cn))
