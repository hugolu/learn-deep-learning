# 深度學習

既然機器學習都摸了，也來玩玩深度學習吧。

話說 [TensorFlow](https://www.tensorflow.org/) 與 [Theano](http://deeplearning.net/software/theano/) 提供的 library 功能強大但不好駕馭，而 [Keras](https://keras.io/) 將兩者包裝成為簡單易用的 library，所以改玩這個。

## 環境
```shell
$ docker pull ermaker/keras
$ docker run --rm -it -v $(pwd)/share:/share ermaker/keras bash
```

```shell
root@d1b8e80cb058:/# ln -s /share/mnist.pkl.gz /root/.keras/datasets/mnist.pkl.gz
```

## Hello world
```shell
root@d1b8e80cb058:~# python /share/keras/examples/mnist_mlp.py
Using TensorFlow backend.
60000 train samples
10000 test samples
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 512)           401920      dense_input_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 512)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 512)           0           activation_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 512)           262656      dropout_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 512)           0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           activation_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            5130        dropout_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 10)            0           dense_3[0][0]
====================================================================================================
Total params: 669706
____________________________________________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 12s - loss: 0.2524 - acc: 0.9211 - val_loss: 0.1234 - val_acc: 0.9614
...(略)
Epoch 20/20
60000/60000 [==============================] - 15s - loss: 0.0227 - acc: 0.9944 - val_loss: 0.1157 - val_acc: 0.9832
Test score: 0.115722299889
Test accuracy: 0.9832
```

## 參考
- [一天搞懂深度學習 心得筆記](https://github.com/hugolu/learning-notes/blob/master/deep-learning.md)
- [keras 官網](https://keras.io/)
- [Keras 中文文档](https://keras-cn.readthedocs.io/en/latest/)
- [ermaker/keras dockerfile](https://hub.docker.com/r/ermaker/keras/)
