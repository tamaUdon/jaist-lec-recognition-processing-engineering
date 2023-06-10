# ref. JAIST 23'認識処理工学特論 lec.2 p.30 
# SciLabによるパーセプトロン学習プログラム例 (1)

import numpy as np

X = np.array([1.2, 0.2, -0.2, -0.5, -1.0, -1.5]) # training data = トレーニンづデータ
y = np.array([1, 1, 1, 2, 2, 2]).T # class of training data = トレーニングデータのクラス
w = np.array([0.5, 0.5]).T # initial weighting coefficients = 重みづけ係数, クラスごとに設定する
r = 0.5 # coeficient of training = トレーニングの係数, rho
flag = True # True/False
n = np.shape(X)[0] # 行数 = 6
d = np.ndim(X) # 次元数 = 1
X = np.stack([np.ones(n), X], 1) # w_0の行数を追加する

print(f"X={X}")
# X=
# [[ 1.   1.2]
#  [ 1.   0.2]
#  [ 1.  -0.2]
#  [ 1.  -0.5]
#  [ 1.  -1. ]
#  [ 1.  -1.5]]

# print(f"[n, d]={[n, d]}")
# [n, d]=[6, 1]

m = 0

while flag:
    flag = False
    m += 1
    print(f"step={m}")

    for i in range(n):
        x = X[i,:].T # Xのi番目の行の全ての要素を取得し転置
        print(f"x={x}")
        g = w.T * x # w=重み係数を転置しxを乗算
        print(f"w={w}")
        print(f"weighting coefficients={w.T}")
        print(f"g={g}")

        if (y[i] == 1) and np.sum(g)<0: # クラス1かどうか判別
            w = w + r*x # トレーニング係数rにx(誤差)を乗算したものを重みwに加算し, wを更新する
            flag = True # 学習継続
            print(f"updated w={w}")
        elif (y[i] == 2) and np.sum(g)>0:# クラス2かどうか判別
            w = w - r*x
            flag = True
            print(f"updated w={w}")
print(f"Results: w0={w[0]}, w1={w[1]}")


