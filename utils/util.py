# coding: utf-8
import numpy as np
from PIL import Image
import math
import sys
sys.path.append('..')


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


#確率を計算
def calc_prob(log_list):
    sum_value = 0
    max_value = math.exp(max(log_list))
    for i in log_list:
        sum_value += math.exp(i) 
    return max_value/sum_value

#画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files,size):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname , size)
    return np.array(X), np.array(Y)

#渡された画像データを読み込んでXに格納し、また、画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname, size):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((size,size))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

def nakamoto_review(ans,a,b,c):
        dic_nakamoto = {}
        dic_nakamoto["gomoku_miso"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:たっぷりの肉と野菜を秘伝の味噌を絡めて炒めた\nボリューム満点の特製味噌ラーメンです。\n値段:980円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["gomoku_moko"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥\n紹介:蒙古タンメンに冷し味噌ラーメンのスープと肉を加えた\n食べ応えのあるやや辛めの味噌ラーメンです。\n値段:880円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["hiyasi_gomoku"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:冷し五目味噌タンメンに辛子麻婆を加え\n辛さとコクをアップいたしました。\n値段:880円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["hiyasi_miso"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:当店で一番辛いスープです。\n突き抜ける辛さと旨さが病みつきになります。\n値段:800円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["hokkyoku"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:熱々の激辛スープ、辛さを極めた味噌ラーメンです。\n初めての方はご注意ください。\n値段:830円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["hokkyoku_yasai"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:北極ラーメンに味噌タンメンの煮込み野菜をトッピングした\nバランスの良い辛旨ラーメンです。\n値段:930円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["miso"] = "名前:"+a+"\n辛さ:🔥🔥🔥\n紹介:肉、野菜たっぷりで辛さ控えめの味噌味のタンメンです。\n初めての方にオススメです。\n値段:780円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["miso_ran"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:肉、もやし、ゆでたまごがたっぷりの\n辛くてボリュームのある味噌ラーメンです。\n値段:880円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["moko"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥\n紹介:当店人気No.1メニュー。\n味噌タンメンの上に辛子麻婆豆腐がのった逸品です。\n値段:800円\n"+b+","+c+"の可能性もあります。"
        dic_nakamoto["moko_ran"] = "名前:"+a+"\n辛さ:🔥🔥🔥🔥🔥🔥🔥🔥\n紹介:味噌卵麺に辛子麻婆をトッピングした、\nより濃厚で辛い味噌ラーメンです。\n値段:900円\n備考:"+b+","+c+"の可能性もあります。"
        dic_nakamoto["sio"] = "名前:"+a+"\n辛さ:無し\n紹介:炒めた野菜がたっぷり入った\nヘルシーで辛くない塩味のタンメンです。\n値段:750円\n備考:"+b+","+c+"の可能性もあります。"
        return dic_nakamoto[ans]

