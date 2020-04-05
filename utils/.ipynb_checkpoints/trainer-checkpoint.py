# coding: utf-8
import sys
sys.path.append('..')
import numpy
import math
from utils.np import *  # import numpy as np
from utils.optimizer import *
from utils import config
from utils.util import to_cpu, to_gpu

class Trainer:

    def __init__(self, network, x_train, t_train, x_val, t_val ,x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        # optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
        #                         'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        optimizer_class_dict = {'sgd':SGD,  'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        # self.test_acc_list = []
        self.val_acc_list = []

    def train_step(self):
        batch_mask = numpy.random.choice(self.train_size, self.batch_size)
        #batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        # x_val_batch = self.x_val[batch_mask]
        # t_val_batch = self.t_val[batch_mask]
        if config.GPU:
            x_batch = to_gpu(x_batch)
            t_batch = to_gpu(t_batch)
            # x_val_batch = to_gpu(x_val_batch)
            # t_val_batch = to_gpu(t_val_batch)


        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss) 

        # val_loss = self.network.loss(x_val_batch, t_val_batch)
        # self.val_loss_list.append(val_loss) 


    #    if self.verbose: print("train loss:" + str(loss),"current_iter:" + str(self.current_iter))
        
        if self.current_iter % self.iter_per_epoch == 0:
            if self.verbose: print("train loss:" + str(loss),"current_iter:" + str(self.current_iter))
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            # x_test_sample, t_test_sample = self.x_test, self.t_test
            x_val_sample, t_val_sample = self.x_val, self.t_val

             
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                # x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                x_val_sample, t_val_sample = self.x_val[:t], self.t_val[:t]
                if config.GPU:

                    x_train_sample, t_train_sample = to_gpu(x_train_sample),to_gpu(t_train_sample)
                    # x_test_sample,t_test_sample = to_gpu(x_test_sample),to_gpu(t_test_sample)
                    x_val_sample,t_val_sample = to_gpu(x_val_sample),to_gpu(t_val_sample)
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample,self.batch_size)
            # test_acc = self.network.accuracy(x_test_sample, t_test_sample,self.batch_size)
            val_acc = self.network.accuracy(x_val_sample, t_val_sample,self.batch_size)
            self.train_acc_list.append(train_acc)
            # self.test_acc_list.append(test_acc)
            self.val_acc_list.append(val_acc)
            

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(val_acc) + " ===")
        self.current_iter += 1

    def train(self):
        print(self.iter_per_epoch)
        if self.verbose: print("max_iter:" + str(self.max_iter))
        for i in range(self.max_iter):
            self.train_step()
        
        x_test,t_test = self.x_test,self.t_test

        test_acc_list = []

        #分割してaccを計算
        #testとvalidationを分ける

        div = 2
        th = len(t_test)//div
        print(len(t_test))

        for i in range(0,div):
            print(i)
            x_test1,t_test1 = x_test[th*i:th*(i+1)],t_test[th*i:th*(i+1)]
            print(th)
            if config.GPU:
                x_test1,t_test1 = to_gpu(x_test1),to_gpu(t_test1)
            test_acc1 = self.network.accuracy(x_test1, t_test1,self.batch_size)
            test_acc_list.append(test_acc1)
            print(test_acc1)

        test_acc = sum(test_acc_list)/div


        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))






