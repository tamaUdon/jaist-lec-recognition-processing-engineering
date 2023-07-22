# CLAFIC法でirisとlfw_peopleを識別
import numpy as np
import pandas as pd
import typing
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris,fetch_lfw_people


# 部分空間法
def clafic(data: pd.DataFrame, labels: np.ndarray) -> None:
    # データ数
    N = data.shape[0]

    # ラベル数
    C = np.unique(labels).shape[0]
    print(f'ラベル数={C}')
    print(f'+++++++++++labels={labels}+++++++++++++++++++')

    # 次元数
    D = data.shape[1]

    # 各画像の固有ベクトルを読み込み
    l,w,f = eigen_vector(data)
    ### Output: 
    # l=[4.20005343 0.24105294 0.0776881  0.02367619]
    # w=[[ 0.36138659 -0.65658877 -0.58202985  0.31548719]
    # [-0.08452251 -0.73016143  0.59791083 -0.3197231 ]
    # [ 0.85667061  0.17337266  0.07623608 -0.47983899]
    # [ 0.3582892   0.07548102  0.54583143  0.75365743]]

    A = w

    # CLAFIC法
    data = data.values # numpy.ndarrayを得る
    shape = C
    out = np.empty(shape)
    for d in range(0, 4):
        count_ok = 0
        for i in range(N): # データ数
            for j in range(shape): # ラベル数
                norm = 0
                for k in range(d): # 
                    # print(f'---------j={j}----------')
                    # print(f'---------k={k}----------')
                    # print(f'---------A={A}----------')
                    # print(f'---------out={out}----------')
                    # print(f'---------data={data}----------')
                    norm += np.dot(data[i,:], A[j,:])**2
                    # print(f'---------norm={norm}----------')
                out[j] = norm
            # print(f'---------out={out}----------')
            print(f'-------------np.argmax(out)={np.argmax(out,axis=0)}-----------------')
            print(f'-------------labels[i]={labels[i]}----------------')
            if (np.argmax(out, axis=0) == labels[i]):
                count_ok += 1

        # 正解率
        print(count_ok/N * 100, d)

# 固有ベクトルを算出
def eigen_vector(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 固有値と固有ベクトルを求める

    X = X.values
    mean = np.mean(X, axis=0)
    X = X - mean

    lowerdim = 2 # 任意の次元へ低次元化
    cov = np.cov(X.T, bias=1) #分散共分散行列
    L, V = np.linalg.eig(cov) # 固有値問題
    inds = np.argsort(L)[::-1] # 固有値の降順ソート
    L = L[inds]
    W = V[:, inds]
    F = np.matmul(X, W[:,:lowerdim]) #主成分得点

    print(f'###########X={X}##############')
    print(f'###########L={L}##############')
    print(f'###########W={W}##############')
    print(f'###########F[0]={F[0]}##############')
    print(f'###########F[1]={F[1]}##############')

    # fig = plt.figure()
    # plt.scatter(F[0,:], F[1,:])
    # plt.grid()
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # fig.savefig("img.png")
    # plt.show()

    return (L,W,F)


def main() -> None:
    # データ読み込み
    iris = load_iris()
    people = fetch_lfw_people()

    # iris
    # df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    #  df_iris['species'] = iris.target_names[iris.target]
    # print(df_iris.head(10))

    X = pd.DataFrame(iris.data,
                 columns=iris.feature_names)

    # # lfw
    # df_people = pd.DataFrame(people.data, columns=people.target)
    # data_lfw = df_people.values
    # print(df_people.head(10))
    
    labels = iris.target
    clafic(X, labels)

if __name__ == "__main__":
    main()
