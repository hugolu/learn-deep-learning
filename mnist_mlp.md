# 訓練簡單的 DNN (deep neuron networks) 辨識 MNIST 資料集

- script: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
- dataset: https://s3.amazonaws.com/img-datasets/mnist.pkl.gz

## MNIST 資料集
包含 6 萬個訓練資料，與 1 萬個測試資料。每筆測試資料為 28x28 pixel 的灰階圖片，對應 0~9 的數字。

用法：
```python
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

回傳兩個 tuple：
- `X_train, X_test`: uint8 array of grayscale image data with shape `(nb_samples, 28, 28)`.
- `y_train, y_test`: uint8 array of digit labels (integers in range 0-9) with shape `(nb_samples,)`.

例如： X_train[1] 長下面這樣(每個0~255的數字對應一個灰階的pixel)，經過人工判讀 y_train[1] 為 0
![](pictures/X_train[1].png)

## 使用 DNN 辨識 MNIST 資料集
source: https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

### 準備訓練/測試資料集
```python
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 載入資料集

X_train = X_train.reshape(60000, 784)   # 處理 60000 筆訓練資料集, 將每筆資料從 28x28 的陣列轉換為 784x1 的陣列
X_test = X_test.reshape(10000, 784)     # 處理 10000 筆測試資料集, 將每筆資料從 28x28 的陣列轉換為 784x1 的陣列
X_train = X_train.astype('float32')     # 處理訓練資料集, 將陣列元素由 uint8 轉型為 float32
X_test = X_test.astype('float32')       # 處理測試資料集, 將陣列元素由 uint8 轉型為 float32
X_train /= 255                          # 處理訓練資料集, 調整陣列元素值範圍，使其介於 0 與 1 之間
X_test /= 255                           # 處理測試資料集, 調整陣列元素值範圍，使其介於 0 與 1 之間
```

### 準備訓練/測試結果
```python
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)  # 對訓練資料集做 1-of-k coding
Y_test = np_utils.to_categorical(y_test, nb_classes)    # 對訓測試料集做 1-of-k coding
```

1-of-k 編碼: 假設變量可取的值有k種可能，如果對這些值用 1 到 k 的編碼，則可用長度為 k 的二元向量來表示一個變量的值。這個向量裡，取值對應的序號所在的元素為1，其餘元素為0。

例如，Y_train[1] 表示對應的圖片被辨識為 0。
```python
>>> Y_train[1]
array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
```

> 如果圖片內容不是數值 0~9，而是 apple, banana, cherry...，輸出結果無法用數值表示，必須通過 1-of-k coding 將類別特徵表示為數字形式。

### 定義模型
```python
model = Sequential()                        # 使用 Sequential model：多層網路的線性堆疊

model.add(Dense(512, input_shape=(784,)))   # 增加全連接的網路層，輸出維度 512, 輸入維度 784
model.add(Activation('relu'))               # 對輸出層施加激活函數 ReLU
model.add(Dropout(0.2))                     # 對輸入層施加 dropout

model.add(Dense(512))                       # 增加全連接的網路層，輸出維度 512
model.add(Activation('relu'))               # 對輸出層施加激活函數 ReLU
model.add(Dropout(0.2))                     # 對輸入層施加 dropout

model.add(Dense(10))                        # 增加全連接的網路層，輸出維度 10
model.add(Activation('softmax'))            # 對輸出層施加激活函數 Softmax
```
- [`Sequential`](https://keras.io/getting-started/sequential-model-guide/): The Sequential model is a linear stack of layers.
- [`Dense`](https://keras.io/layers/core/#dense): Just your regular fully connected NN layer.
- [`Activation`](https://keras.io/layers/core/#activation): Applies an activation function to an output.
- [`Dropout`](https://keras.io/layers/core/#dropout): Applies Dropout to the input. Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting.

### 編譯模型
```python
model.compile(loss='categorical_crossentropy',        # 設定損失函數，評估準確度
              optimizer=RMSprop(),                    # 設定 optimizer，決定學習速度
              metrics=['accuracy'])                   # 設定 metrics，評斷模型效能
```
- `loss` is the objective that the model will try to minimize.
- `optimizer` is the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the  Optimizer class.
- `metrics`: For any classification problem you will want to set this to metrics=['accuracy'].

### 訓練模型
```python
history = model.fit(X_train,                          # 輸入資料
                    Y_train,                          # 標籤
                    batch_size=batch_size,            # 進行梯度下降(SGD)，每個 batch 包含的樣本數
                    nb_epoch=nb_epoch,                # 訓練幾輪
                    verbose=1,                        # 顯示訓練過程
                    validation_data=(X_test, Y_test)) # 驗證集
```

### 評估模型
```python
score = model.evaluate(X_test, Y_test, verbose=0)     # 使用驗證集為模型打分數
print('Test score:', score[0])                        # Test score: 0.124787330822
print('Test accuracy:', score[1])                     # Test accuracy: 0.9816
```
