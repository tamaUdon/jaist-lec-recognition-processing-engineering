# ref. JAIST 23'認識処理工学特論 lec.2 レポートプログラム
# Pythonによるパーセプトロン学習プログラム例 (重み固定・変動増分手法)

import numpy as np

X = np.array([1.2, 0.2, -0.2, -0.5, -1.0, -1.5]) # training data = トレーニンづデータ
y = np.array([1, 1, 1, 2, 2, 2]).T # class of training data = トレーニングデータのクラス
w = 0.5 # 固定値. initial weighting coefficients = 重みづけ係数, クラスごとに設定する
r = np.array([0.5, 0.5]).T # 変動値. coeficient of training = トレーニングの初期係数, rho
flag = True # True/False
n = np.shape(X)[0] # 行数 = 6
d = np.ndim(X) # 次元数 = 1
X = np.stack([np.ones(n), X], 1) # rの行数を追加する

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
        # g = w.T * x # w=重み係数を転置しxを乗算
        g = r.T * x # r=重み係数の移動量にxを乗算
        print(f"r={r}")
        print(f"weighting coefficients={r.T}")
        print(f"g={g}")

        if (y[i] == 1) and np.sum(g)<0: # クラス1かどうか判別
            # w = w + r*x # トレーニング係数rにx(誤差)を乗算したものを重みwに加算し, wを更新する
            r = r + w*x
            flag = True # 学習継続
            print(f"updated r={r}")
        elif (y[i] == 2) and np.sum(g)>0:# クラス2かどうか判別
            # w = w - r*x
            r = r - w*x
            flag = True
            print(f"updated r={r}")
print(f"Results: r0={r[0]}, r1={r[1]}")


