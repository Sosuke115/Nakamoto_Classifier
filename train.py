# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポート
from PIL import Image
import os, glob
import numpy as np
import random, math
from utils import config
import time
from utils.trainer import Trainer
from utils.util import calc_prob,make_sample,add_sample
import argparse
# config.GPU = True
# import cupy as cp
# from common.util import to_cpu, to_gpu

####jupyternotebook用、グラフ描画
import matplotlib
# %matplotlib inline
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#####

#networkの選択

from models.alexnet import AlexNet
from models.deepsimplenet import DeepSimpleNet
from models.vgg11 import VGG11



def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file_path', type=str, default="image_data",
                    help='train data path')
    parser.add_argument('--reshape_size', type=int, default=227,
                    help='reshape size')
    parser.add_argument('--model', type=str, default="alexnet",
                    help='alexnet or vgg11 or deepsimplenet')
    args = parser.parse_args()
    print("Train Start")
    
    # root_dir = "dataset_nakamoto_inf" 
    train_dir = args.file_path


    #カテゴリーを指定

    categories = ["gomoku_miso","gomoku_moko","hiyasi_gomoku"
              ,"hiyasi_miso","hokkyoku","hokkyoku_yasai"
             ,"miso","miso_ran","moko","moko_ran","sio"]

    t1 = time.time() 

    # 画像データ用配列
    X = []
    # ラベルデータ用配列
    Y = []
    #全データ格納用配列
    allfiles = []

    #カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
    for idx, item in enumerate(categories):
        image_dir = train_dir + "/" + item
        files = glob.glob(image_dir + "/*")
        # file種類に合わせて変更
        # files = glob.glob(image_dir + "/*.png")
    #     files = glob.glob(image_dir + "/*.jpg")
        # files = glob.glob(image_dir + "/*.jpeg")
        for f in files:
            allfiles.append((idx, f))

    #シャッフル後、学習データと検証データに分ける
    random.shuffle(allfiles)
    th = math.floor(len(allfiles) * 0.8)
    train = allfiles[0:th]
    test  = allfiles[th:]
    X_train, y_train = make_sample(train,args.reshape_size)
    X_test, y_test = make_sample(test,args.reshape_size)


    X_train = X_train.astype(np.float32)
    X_train /= 255.0

    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    #testとvalidationを分ける
    th = math.floor(len(y_test) * 0.5)
    X_val, y_val = X_test[:th], y_test[:th]
    X_test, y_test = X_test[th:], y_test[th:]

    x_train, t_train = X_train.transpose(0,3,1,2), y_train
    x_val, t_val = X_val.transpose(0,3,1,2), y_val
    x_test, t_test = X_test.transpose(0,3,1,2), y_test
    
    max_epochs = 10

    """
    if config.GPU:
        x_train, t_train = to_gpu(x_train), to_gpu(t_train)
        x_test, t_test = to_gpu(x_test), to_gpu(t_test)
    """


    if args.model == "alexnet":
        network = AlexNet(input_dim=(3, args.reshape_size, args.reshape_size),output_size = len(categories))
    elif args.model == "vgg11":
        network = AlexNet(input_dim=(3, args.reshape_size, args.reshape_size),output_size = len(categories))       
    elif args.model == "deepsimplenet":
        network = AlexNet(input_dim=(3, args.reshape_size, args.reshape_size),output_size = len(categories))
        
    #訓練開始
    trainer = Trainer(network, x_train, t_train, x_val, t_val,x_test, t_test,
                  epochs=max_epochs, mini_batch_size=10,
                  optimizer='Adam', optimizer_param={'lr': 0.0001},
                  evaluate_sample_num_per_epoch=50
                  )

    trainer.train()

    # パラメータの保存
    #{名}_{network名}.pkl
    network.save_params("results/test" + args.model + ".pkl")
    print("Saved Network Parameters!")


    t2 = time.time()
    elapsed_time = t2-t1
    print(f"time：{elapsed_time}")

    #訓練結果を描画

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)

    ###できれば複数グラフにする
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.val_acc_list, marker='s', label='validation', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.title("Training and validation accuracy")
    # plt.show()
    plt.savefig("results/accuracy.png")


if __name__ == "__main__":
    main()



