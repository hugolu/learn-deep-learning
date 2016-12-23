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
- `X_train, X_test`: uint8 array of grayscale image data with shape (nb_samples, 28, 28).
- `y_train, y_test`: uint8 array of digit labels (integers in range 0-9) with shape (nb_samples,).
