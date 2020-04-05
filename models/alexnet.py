# coding: utf-8
# import sys, os
# sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import sys
sys.path.append('..')
import pickle
# import numpy as np
from utils import config
from utils.util import to_cpu, to_gpu
from utils.np import *
from collections import OrderedDict
from utils.layers import *
import math


prob = []
class AlexNet:
    """

input			227,227,3
conv	11,11	4,4	55,55,96
maxpool	3,3	2,2	27,27,96
conv	5,5	1,1	13,13,256
maxpool	3,3	2,2	13,13,256
conv	3,3	1,1	13,13,384
conv	3,3	1,1	13,13,384
conv	3,3	1,1	13,13,256
maxpool	3,3	2,2	6,6,256
fc			4096
fc			4096
fc          1000



        conv - relu - pool- conv - relu - pool -
        conv - relu - conv- relu - conv - relu -
        pool - affine - relu - affine - relu -dropout -affine dropout softmax
        0, 3, 6, 8, 10, 13, 15, 18
    """

    def __init__(self, input_dim=(3, 227, 227),
                 conv_param_1 = {'filter_num':96, 'filter_size':11, 'pad':0, 'stride':4},
                 conv_param_2 = {'filter_num':256, 'filter_size':5, 'pad':0, 'stride':1},
                 conv_param_3 = {'filter_num':384, 'filter_size':3, 'pad':0, 'stride':1},
                 conv_param_4 = {'filter_num':384, 'filter_size':3, 'pad':0, 'stride':1},
                 conv_param_5 = {'filter_num':256, 'filter_size':3, 'pad':0, 'stride':1},
                #  conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=1000, output_size=11):
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
        pre_node_nums = np.array([input_dim[0]*3*3, 96*11*11, 256*5*5, 284*3*3, 384*3*3, 384*3*3, 256*3*3, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値
        
        self.params = {}

        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W6'] = weight_init_scales[5] * np.random.randn(1024, 4096)
        self.params['b6'] = np.zeros(4096)
        self.params['W7'] = weight_init_scales[6] * np.random.randn(4096, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

                # レイヤの生成===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=3, pool_w=3, stride=2))
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=3, pool_w=3, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=3, pool_w=3, stride=2))
        self.layers.append(Affine(self.params['W6'], self.params['b6']))
        self.layers.append(Relu())
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
            u = 0
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 3, 6, 8, 10, 13, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            if config.GPU:
                val = to_cpu(val)
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 3, 6, 8, 10, 13, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]








    