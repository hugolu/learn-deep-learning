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

...(略)
Test score: 0.115722299889
Test accuracy: 0.9832
```

## 牛刀小試
試著使用 DNN 與 CNN 進行圖形辨識，實驗過程筆記在 [辨識 MNIST 資料集](mnist.md)。

## 參考
- [一天搞懂深度學習 心得筆記](https://github.com/hugolu/learning-notes/blob/master/deep-learning.md)
- [keras 官網](https://keras.io/)
- [Keras 中文文档](https://keras-cn.readthedocs.io/en/latest/)
- [ermaker/keras dockerfile](https://hub.docker.com/r/ermaker/keras/)
