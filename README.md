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
Epoch 2/20
60000/60000 [==============================] - 12s - loss: 0.1069 - acc: 0.9673 - val_loss: 0.0797 - val_acc: 0.9757
Epoch 3/20
60000/60000 [==============================] - 12s - loss: 0.0795 - acc: 0.9769 - val_loss: 0.0769 - val_acc: 0.9777
Epoch 4/20
60000/60000 [==============================] - 13s - loss: 0.0627 - acc: 0.9809 - val_loss: 0.0762 - val_acc: 0.9790
Epoch 5/20
60000/60000 [==============================] - 13s - loss: 0.0523 - acc: 0.9848 - val_loss: 0.0901 - val_acc: 0.9783
Epoch 6/20
60000/60000 [==============================] - 13s - loss: 0.0474 - acc: 0.9852 - val_loss: 0.0776 - val_acc: 0.9804
Epoch 7/20
60000/60000 [==============================] - 14s - loss: 0.0416 - acc: 0.9877 - val_loss: 0.0742 - val_acc: 0.9816
Epoch 8/20
60000/60000 [==============================] - 13s - loss: 0.0378 - acc: 0.9891 - val_loss: 0.0783 - val_acc: 0.9834
Epoch 9/20
60000/60000 [==============================] - 13s - loss: 0.0328 - acc: 0.9908 - val_loss: 0.0970 - val_acc: 0.9801
Epoch 10/20
60000/60000 [==============================] - 12s - loss: 0.0331 - acc: 0.9904 - val_loss: 0.0868 - val_acc: 0.9821
Epoch 11/20
60000/60000 [==============================] - 13s - loss: 0.0312 - acc: 0.9914 - val_loss: 0.0816 - val_acc: 0.9830
Epoch 12/20
60000/60000 [==============================] - 12s - loss: 0.0275 - acc: 0.9922 - val_loss: 0.1008 - val_acc: 0.9808
Epoch 13/20
60000/60000 [==============================] - 14s - loss: 0.0272 - acc: 0.9923 - val_loss: 0.1014 - val_acc: 0.9803
Epoch 14/20
60000/60000 [==============================] - 12s - loss: 0.0292 - acc: 0.9922 - val_loss: 0.1015 - val_acc: 0.9826
Epoch 15/20
60000/60000 [==============================] - 13s - loss: 0.0249 - acc: 0.9938 - val_loss: 0.1007 - val_acc: 0.9816
Epoch 16/20
60000/60000 [==============================] - 16s - loss: 0.0234 - acc: 0.9936 - val_loss: 0.1033 - val_acc: 0.9816
Epoch 17/20
60000/60000 [==============================] - 13s - loss: 0.0250 - acc: 0.9939 - val_loss: 0.1118 - val_acc: 0.9830
Epoch 18/20
60000/60000 [==============================] - 13s - loss: 0.0239 - acc: 0.9938 - val_loss: 0.0934 - val_acc: 0.9835
Epoch 19/20
60000/60000 [==============================] - 14s - loss: 0.0215 - acc: 0.9948 - val_loss: 0.1198 - val_acc: 0.9815
Epoch 20/20
60000/60000 [==============================] - 15s - loss: 0.0227 - acc: 0.9944 - val_loss: 0.1157 - val_acc: 0.9832
Test score: 0.115722299889
Test accuracy: 0.9832
```

## 參考
- [一天搞懂深度學習 心得筆記](https://github.com/hugolu/learning-notes/blob/master/deep-learning.md)
- [keras 官網](https://keras.io/)
- [ermaker/keras dockerfile](https://hub.docker.com/r/ermaker/keras/)
