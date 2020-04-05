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
from utils.util import calc_prob,make_sample,add_sample,nakamoto_review
from colorama import Fore, Back, Style
import argparse
# config.GPU = True
# import cupy as cp
# from common.util import to_cpu, to_gpu

#networkの選択

from models.alexnet import AlexNet
from models.deepsimplenet import DeepSimpleNet
from models.vgg11 import VGG11


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--weight_path', type=str, default="results/nakamoto_dataaug_alexnet.pkl",
                    help='weight data path')
    parser.add_argument('--reshape_size', type=int, default=227,
                    help='reshape size')
    parser.add_argument('--model', type=str, default="alexnet",
                    help='alexnet or vgg11 or deepsimplenet')
    args = parser.parse_args()
    print("judge start")

    #中本
    categories = ["gomoku_miso","gomoku_moko","hiyasi_gomoku"
                ,"hiyasi_miso","hokkyoku","hokkyoku_yasai"
                ,"miso","miso_ran","moko","moko_ran","sio"]

    categories_ja = ["五目味噌タンメン","五目蒙古タンメン","冷やし五目"
                ,"冷やし味噌野菜","北極","北極野菜","味噌タンメン"
                ,"味噌卵麺","蒙古タンメン","蒙古卵麺","塩タンメン"]
    
    #pokemon
    # categories = ["pikachu","Eevee","Numera"]

    
    # root_dir = "dataset_nakamoto_inf" 
    judge_data = ["judge_data"]

    prob = []
    # 画像データ用配列
    X = []
    # ラベルデータ用配列
    Y = []

    if args.model == "alexnet":
        network = AlexNet(input_dim=(3, args.reshape_size, args.reshape_size),output_size = len(categories))
    elif args.model == "vgg11":
        network = AlexNet(input_dim=(3, args.reshape_size, args.reshape_size),output_size = len(categories))       
    elif args.model == "deepsimplenet":
        network = AlexNet(input_dim=(3, args.reshape_size, args.reshape_size),output_size = len(categories))
    
    # network.load_params("nakamotoinf_alexnet.pkl")
    network.load_params(args.weight_path)

    #全データ格納用配列
    allfiles = []

    #カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
    for idx, item in enumerate(judge_data):
        image_dir = item
        files = glob.glob(image_dir + "/*")
        for f in files:
            allfiles.append((idx, f))

    X_train, y_train = make_sample(allfiles,args.reshape_size)
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    x_train = X_train.transpose(0,3,1,2)

    print(Fore.RED)

    y = network.predict(x_train)

    for i in range(len(y)):
        print("")
        print("")
        print("")
        answer = y[i].argsort()[::-1]
        print("予測精度:",int(calc_prob(y[i])*100),"%")
        print(nakamoto_review(categories[answer[0]],categories_ja[answer[0]],categories_ja[answer[1]],categories_ja[answer[2]]))

    print(Style.RESET_ALL)

if __name__ == "__main__":
    main()

