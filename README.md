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
439/439 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 0.0129 - mean_absolute_error: 0.0789
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
Input shape: (32, 24, 19)
Output shape: (32, 24, 1)
```
透過繪製基線模型的預測值，可以注意到只是標籤向右移動了1 小時：
```
wide_window.plot(baseline)
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%92%90%E9%9B%8616.png)

在上面三個樣本的繪圖中，單步驟模型運行了24 小時。這需要一些解釋：

● 藍色的Inputs行顯示每個時間步驟的輸入溫度。模型會接收所有特徵，而該繪圖僅顯示溫度。

● 綠色的Labels點顯示目標預測值。這些點在預測時間，而不是輸入時間顯示。這就是為什麼標籤範圍相對於輸入移動了1 步。

● 橘色的Predictions叉是模型針對每個輸出時間步驟的預測。如果模型能夠進行完美預測，則預測值將直接落在Labels上。

### 線性模型
可以應用於此任務的最簡單的可訓練模型是在輸入和輸出之間插入線性轉換。在這種情況下，時間步驟的輸出僅取決於該步驟：

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E5%AE%A4%E7%AA%97%E5%8C%966.png)

沒有設定activation的tf.keras.layers.Dense層是線性模型。圖層僅會將資料的最後一個軸從(batch, time, inputs)轉換為(batch, time, units)；它會單獨應用於batch和time軸的每個條目。
```
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
```
```
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)
```
```
Input shape: (32, 1, 19)
Output shape: (32, 1, 1)
```
本教學訓練許多模型，因此將訓練過程打包到一個函數中：
```
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history
```
訓練模型並評估其表現：
```
history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
```
```
Epoch 1/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.6573 - mean_absolute_error: 0.5335 - val_loss: 0.0140 - val_mean_absolute_error: 0.0884
Epoch 2/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - loss: 0.0133 - mean_absolute_error: 0.0863 - val_loss: 0.0117 - val_mean_absolute_error: 0.0813
Epoch 3/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 12s 5ms/step - loss: 0.0114 - mean_absolute_error: 0.0797 - val_loss: 0.0100 - val_mean_absolute_error: 0.0744
Epoch 4/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 13s 6ms/step - loss: 0.0100 - mean_absolute_error: 0.0739 - val_loss: 0.0091 - val_mean_absolute_error: 0.0708
Epoch 5/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - loss: 0.0093 - mean_absolute_error: 0.0712 - val_loss: 0.0089 - val_mean_absolute_error: 0.0702
Epoch 6/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - loss: 0.0091 - mean_absolute_error: 0.0702 - val_loss: 0.0088 - val_mean_absolute_error: 0.0689
Epoch 7/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 13s 5ms/step - loss: 0.0090 - mean_absolute_error: 0.0697 - val_loss: 0.0088 - val_mean_absolute_error: 0.0692
Epoch 8/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - loss: 0.0090 - mean_absolute_error: 0.0698 - val_loss: 0.0087 - val_mean_absolute_error: 0.0690
Epoch 9/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - loss: 0.0090 - mean_absolute_error: 0.0695 - val_loss: 0.0087 - val_mean_absolute_error: 0.0687
Epoch 10/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 12s 5ms/step - loss: 0.0090 - mean_absolute_error: 0.0697 - val_loss: 0.0087 - val_mean_absolute_error: 0.0685
Epoch 11/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 12s 6ms/step - loss: 0.0090 - mean_absolute_error: 0.0696 - val_loss: 0.0086 - val_mean_absolute_error: 0.0683
Epoch 12/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - loss: 0.0090 - mean_absolute_error: 0.0697 - val_loss: 0.0087 - val_mean_absolute_error: 0.0683
Epoch 13/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 6s 4ms/step - loss: 0.0090 - mean_absolute_error: 0.0698 - val_loss: 0.0087 - val_mean_absolute_error: 0.0687
439/439 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - loss: 0.0085 - mean_absolute_error: 0.0682
```
與baseline模型類似，可以在寬度視窗的批次上呼叫線性模型。使用這種方式，模型會在連續的時間步驟上進行一系列獨立預測。time軸的作用類似另一個batch軸。在每個時間步驟上，預測之間沒有交互作用。

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E5%AE%A4%E7%AA%97%E5%8C%969.png)

```
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
```
Input shape: (32, 24, 19)
Output shape: (32, 24, 1)
```
下面是wide_widow上它的樣本預測圖。請注意，在許多情況下，預測值顯然比僅返回輸入溫度更好，但在某些情況下則會更差：****
```
```
wide_window.plot(linear)
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%92%90%E9%9B%8612.png)

線性模型的優點之一是它們相對易於解釋。您可以拉取層的權重，並呈現分配給每個輸入的權重：

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%92%90%E9%9B%8613.png)

```
plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
```
有時模型甚至不會將大多數權重放在輸入T (degC)上。這是隨機初始化的風險之一。

### 密集
在應用實際運算多個時間步驟的模型之前，值得先研究一下更深、更強大的單輸入步驟模型的表現。

下面是一個與linear模型類似的模型，只不過它在輸入和輸出之間堆疊了幾個Dense層：
```
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
```
```
Epoch 1/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - loss: 0.0420 - mean_absolute_error: 0.1111 - val_loss: 0.0081 - val_mean_absolute_error: 0.0659
Epoch 2/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - loss: 0.0079 - mean_absolute_error: 0.0642 - val_loss: 0.0079 - val_mean_absolute_error: 0.0652
Epoch 3/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 9s 6ms/step - loss: 0.0075 - mean_absolute_error: 0.0624 - val_loss: 0.0082 - val_mean_absolute_error: 0.0661
Epoch 4/20
1534/1534 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - loss: 0.0073 - mean_absolute_error: 0.0612 - val_loss: 0.0071 - val_mean_absolute_error: 0.0610
```
### 多步驟密集
單時間步驟模型沒有其輸入的目前值的上下文。它看不到輸入特徵隨時間變化的情況。要解決此問題，模型在進行預測時需要存取多個時間步驟：

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E5%AE%A4%E7%AA%97%E5%8C%9610.png)

baseline、linear和dense模型會單獨處理每個時間步驟。在這裡，模型將接受多個時間步驟作為輸入，以產生單一輸出。

建立一個WindowGenerator，它將產生3 小時輸入和1 小時標籤的批次：

請注意，Window的shift參數與兩個視窗的末端相關。

```
CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'])

conv_window
```
```
Total window size: 4
Input indices: [0 1 2]
Label indices: [3]
Label column name(s): ['T (degC)']
```
```
conv_window.plot()
plt.title("Given 3 hours of inputs, predict 1 hour into the future.")
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%92%90%E9%9B%8614.png)

您可以透過新增tf.keras.layers.Flatten作為模型的第一層，在多輸入步驟視窗上訓練dense模型：
```
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
```
```
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
```
```
Input shape: (32, 3, 19)
Output shape: (32, 1, 1)
```
```
history = compile_and_fit(multi_step_dense, conv_window)

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0) 
```
```
conv_window.plot(multi_step_dense)
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%92%90%E9%9B%8615.png)

此方法的主要缺點是，產生的模型只能在具有此形狀的輸入視窗上執行。

```
print('Input shape:', wide_window.example[0].shape)
try:
  print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')
```
```
Input shape: (32, 24, 19)

ValueError:Exception encountered when calling Sequential.call().

Input 0 of layer "dense_4" is incompatible with the layer: expected axis -1 of input shape to have value 57, but received input with shape (32, 456)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(32, 24, 19), dtype=float32)
  • training=None
  • mask=None
```
下一部分的捲積模型將解決這個問題。

### 卷積神經網絡
卷積層( tf.keras.layers.Conv1D) 也需要多個時間步驟作為每個預測的輸入。

下面的模型與multi_step_dense 相同，使用卷積進行了重寫。

請注意以下變化：

tf.keras.layers.Flatten和第一個tf.keras.layers.Dense替換成了tf.keras.layers.Conv1D。
由於卷積將時間軸保留在其輸出中，不再需要tf.keras.layers.Reshape
```
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])
```
```
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)
```
```
Conv model on `conv_window`
Input shape: (32, 3, 19)
Output shape: (32, 1, 1)
```
在一個樣本批次上運行上述模型，以查看模型是否產生了具有預期形狀的輸出：
```
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)
```
```
Wide window
Input shape: (32, 24, 19)
Labels shape: (32, 24, 1)
Output shape: (32, 22, 1)
```
在conv_window上訓練和評估上述模型，它應該提供與multi_step_dense模型類似的性能。
```
history = compile_and_fit(conv_model, conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
```
此conv_model和multi_step_dense模型的差異在於，conv_model可以在任意長度的輸入上運行。卷積層應用於輸入的滑動視窗：

![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E8%B3%87%E6%96%99%E5%AE%A4%E7%AA%97%E5%8C%9610.png)

```
print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)
```
```
Wide window
Input shape: (32, 24, 19)
Labels shape: (32, 24, 1)
Output shape: (32, 22, 1)
```
請注意，輸出比輸入短。要進行訓練或繪圖，需要標籤和預測具有相同長度。因此，建置WindowGenerator以使用一些額外輸入時間步驟產生寬窗口，從而使標籤和預測長度匹配：
```
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['T (degC)'])

wide_conv_window
```
```
Total window size: 27
Input indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25]
Label indices: [ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
Label column name(s): ['T (degC)']
```
```
print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)
```
```
Wide window
Input shape: (32, 24, 19)
Labels shape: (32, 24, 1)
Output shape: (32, 22, 1)
```
現在，您可以在更寬的視窗上繪製模型的預測。請注意第一個預測之前的3 個輸入時間步驟。這裡的每個預測都基於之前的3 個時間步驟：
```
wide_window.plot(conv_model)
```
![image](https://github.com/Roseller37/ai-final-report/blob/main/image/%E5%A4%A9%E6%B0%A3%E8%92%90%E9%9B%8617.png)

### 循環神經網絡
循環神經網路(RNN) 是一種非常適合時間序列資料的神經網路。 RNN 逐步處理時間序列，從時間步驟到時間步驟維護內部狀態。

您可以在使用RNN 的文本生成教程和使用Keras 的遞歸神經網路(RNN)指南中了解詳情。

在本教學中，您將使用稱為「長短期記憶網路」( tf.keras.layers.LSTM) 的RNN 層。

對所有Keras RNN 層（例如tf.keras.layers.LSTM）都很重要的一個建構函數參數是return_sequences。此設定可以透過以下兩種方式配置層：

1.如果為False（預設值），則圖層僅傳回最終時間步驟的輸出，使模型有時間在進行單一預測之前對其內部狀態進行預熱：
lstm 預熱並進行單一預測

1.如果為True，層將為每個輸入傳回一個輸出。這對以下情況十分有用：
● 堆疊RNN 層。
● 同時在多個時間步驟上訓練模型。
![image]()
```
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])
```
`return_sequences=True`時，模型一次可以在24 小時的資料上進行訓練。

註：這將對模型的性能給予悲觀看法。在第一個時間步驟中，模型無法存取先前的步驟，因此無法比之前展示的簡單`linear`和`dense`模型表現得更好。
```
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)
```
```
Input shape: (32, 24, 19)
Output shape: (32, 24, 1)
```
```
history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
```
```
438/438 [==============================] - 1s 3ms/step - loss: 0.0057 - mean_absolute_error: 0.0522
```
```
wide_window.plot(lstm_model)
```
![image]()
### 效能
使用此資料集時，通常每個模型的效能都比之前的模型稍好：
```
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
```
![image]()
```
for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')
```
```
Baseline    : 0.0852
Linear      : 0.0663
Dense       : 0.0602
Multi step dense: 0.0560
Conv        : 0.0596
LSTM        : 0.0530
```
### 多重輸出模型
到目前為止，所有模型都為單一時間步驟預測了單一輸出特徵，`T (degC)`。

只需更改輸出層中的單元數並調整訓練窗口，以將所有特徵納入labels( example_labels) 中，即可將所有上述模型轉換為預測多個特徵：
```
single_step_window = WindowGenerator(
    # `WindowGenerator` returns all features as labels if you 
    # don't set the `label_columns` argument.
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
```
```
Inputs shape (batch, time, features): (32, 24, 19)
Labels shape (batch, time, features): (32, 24, 19)
```
請注意，上面標籤的`features`軸現在具有與輸入相同的深度，而不是1。

### 基線
這裡可以使用相同的基準模型( `Baseline`)，但這次重複所有特徵，而不是選擇特定的label_index：
```
baseline = Baseline()
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
```
```
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(wide_window.val)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)
```
```
438/438 [==============================] - 1s 2ms/step - loss: 0.0886 - mean_absolute_error: 0.1589
```
### 密集
```
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])
```
```
history = compile_and_fit(dense, single_step_window)

IPython.display.clear_output()
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
```
```
439/439 [==============================] - 1s 3ms/step - loss: 0.0679 - mean_absolute_error: 0.1310
```
### RNN
```
%%time
wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate( wide_window.val)
performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0)

print()
```
```
438/438 [==============================] - 1s 3ms/step - loss: 0.0613 - mean_absolute_error: 0.1195

CPU times: user 5min 54s, sys: 1min 12s, total: 7min 7s
Wall time: 2min 38s
```
#### 進階：殘差連接
先前的Baseline模型利用了以下事實：序列在時間步驟之間不會劇烈變化。到目前為止，本教程中訓練的每個模型都進行了隨機初始化，然後必須學習輸出相較上一個時間步驟改變較小這一知識。

儘管您可以透過仔細初始化來解決此問題，但將此問題建置到模型結構中則更加簡單。

在時間序列分析中建立的模型，通常會預測下一個時間步驟中的值會如何變化，而不是直接預測下一個值。類似地，深度學習中的[殘差網路](https://arxiv.org/abs/1512.03385)（或ResNet）指的是，每一層都會加入模型的累積結果中的架構。

這就是利用「改變應該較小」這一知識的方式。
![image]()
本質上，這將初始化模型以匹配`Baseline`。對於此任務，它可以幫助模型更快收斂，且效能稍好。

該方法可以與本教程中討論的任何模型結合使用。

這裡將它應用於LSTM 模型，請注意`[tf.initializers.zeros](https://tensorflow.google.cn/api_docs/python/tf/keras/initializers/Zeros)`的使用，以確保初始的預測改變很小，並且不會壓制殘差連接。此處的梯度沒有破壞對稱性的問題，因為`zeros`僅用於最後一層。
```
class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each time step is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta
```
```
%%time
residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small.
        # Therefore, initialize the output layer with zeros.
        kernel_initializer=tf.initializers.zeros())
]))

history = compile_and_fit(residual_lstm, wide_window)

IPython.display.clear_output()
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
print()
```
```
438/438 [==============================] - 1s 3ms/step - loss: 0.0622 - mean_absolute_error: 0.1180

CPU times: user 1min 59s, sys: 23.3 s, total: 2min 23s
Wall time: 54.9 s
```
#### 效能
以下是這些多輸出模型的整體表現。
```
x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
plt.ylabel('MAE (average over all outputs)')
_ = plt.legend()
```
![image]()
```
for name, value in performance.items():
  print(f'{name:15s}: {value[1]:0.4f}')
```
```
Baseline : 0.1638
Dense : 0.1320
LSTM : 0.1209
Residual LSTM : 0.1193
```
以上表現是所有模型輸出的平均值。

### 多步驟模型
前幾個部分中的單輸出和多輸出模型都對未來1 小時進行單一時間步驟預測。

本部分介紹如何擴展這些模型以進行多時間步驟預測。

在多步驟預測中，模型需要學習預測一系列未來值。因此，與單步模型（僅預測單一未來點）不同，多步驟模型預測未來值的序列。

大致上有兩種預測方法：

單次預測，一次預測整個時間序列。
自迴歸預測，模型僅進行單步預測並將輸出作為輸入進行回饋。
在本部分中，所有模型都將預測所有輸出時間步驟中的所有特徵。

對於多步驟模型而言，訓練資料仍由每小時樣本組成。但是，在這裡，模型將在給定過去24 小時的情況下學習預測未來24 小時。

下面是一個`Window`對象，該對象從資料集產生以下切片：
```
```
```
```
```
```
![image]()
# 參考資料
[時間序列預測]([https://www.cc.ntu.edu.tw/chinese/epaper/0052/20200320_5207.html](https://tensorflow.google.cn/tutorials/structured_data/time_series?hl=zh_cn))
