### 検証環境: 
# macOS Montray ver.12.2.1
# model: MacBook Pro (13-inch, 2020, Four Thunderbolt 3 ports)
# cpu: 2 GHz クアッドコアIntel Core i5
# memory: 16 GB 3733 MHz LPDDR4X
# python 3.11.3
# poetry 1.4.2

### Installation: 
# $ poetry install
# $ poetry run python lec2_report.py

### 検証手順
# svm_with_no_fine_tune_and_split_direct() ... ファインチューニングなしSVM + シャッフルせず分割
# svm_with_no_fine_tune_and_split_random() ... ファインチューニングなしSVM + シャッフル分割
# fine_tune_grid_search_and_k_fold() ... グリッドサーチ + k分割交差検証
# fine_tune_random_search_and_k_fold() ... ランダムサーチ + k分割交差検証
# 
# 以上3つを実行し比較した結果
# ファインチューニングなしSVM + シャッフル分割の方が、ファインチューニングなしSVM + シャッフルせず分割より0.315789473684211ポイント高い精度を記録した
# グリッドサーチ + k分割交差検証の方が、ファインチューニングなしSVM + 手動分割より0.018847578947369ポイント高い精度となるハイパーパラメータを発見した
# ランダムサーチ + k分割交差検証の方が、グリッドサーチ + k分割交差検証より0.007117ポイント高い精度となるハイパーパラメータを発見した
# よって、データ全体を偏りなく学習させること、学習時のハイパーパラメータ調整の試行回数を増やすことが "賢い"ハイパーパラメータ(rho)の決定方法の一つであることがわかった。

import pandas as pd
import numpy as np
import time

from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from multiprocessing import cpu_count

iris = load_iris()

### svm with no fine tuning + split shuffle=False ###
def svm_with_no_fine_tune_and_split_direct():
    # shuffle=False split
    X_train, X_test = train_test_split(iris.data, shuffle=False)
    y_train, y_test = train_test_split(iris.target, shuffle=False)

    clf = SVC() # no parameter set
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    print("finished: {:.2f} sec".format(t2 - t1))
    acc = np.mean(y_test == clf.predict(X_test))  # manual validation
    print(acc)

    ### results:
    # finished: 0.01 sec
    # 0.631578947368421 <- this is a baseline

### svm with no fine tuning + split random ###
def svm_with_no_fine_tune_and_split_random():
    # random split
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

    clf = SVC() # no parameter set
    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()

    print("finished: {:.2f} sec".format(t2 - t1))
    acc = np.mean(y_test == clf.predict(X_test))  # manual validation
    print(acc)

    ### results:
    # finished: 0.01 sec
    # 0.9473684210526315 <- overfeat svm_with_no_fine_tune_and_split_direct()!

### tuning p by grid search + k-fold ###
def fine_tune_grid_search_and_k_fold():
    svm = SVC(kernel='rbf')
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    candidate_params = {
        'C': [1, 10, 100],
        'gamma': [0.01, 0.1, 1],
    }
    # execute gridsearch and k-fold cross validation by setting 'kf'
    gs = GridSearchCV(estimator=svm, param_grid=candidate_params, cv=kf, n_jobs=cpu_count())

    t1 = time.time()
    gs.fit(iris.data, iris.target) # train
    t2 = time.time()

    print("finished: {:.2f} sec".format(t2 - t1))
    result_df = pd.DataFrame(gs.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    print(result_df[["rank_test_score", 
                     "params", 
                     "mean_test_score"]])
    
    ### results:
    # finished: 4.95 sec
    # rank_test_score                     params  mean_test_score
    # 2                1       {'C': 1, 'gamma': 1}         0.966216 <- overfeat no_fine_tune()!
    # 6                1  {'C': 100, 'gamma': 0.01}         0.966216
    # 1                3     {'C': 1, 'gamma': 0.1}         0.959815
    # 3                3   {'C': 10, 'gamma': 0.01}         0.959815
    # 4                5    {'C': 10, 'gamma': 0.1}         0.959637
    # 5                5      {'C': 10, 'gamma': 1}         0.959637
    # 7                7   {'C': 100, 'gamma': 0.1}         0.953058
    # 0                8    {'C': 1, 'gamma': 0.01}         0.946302
    # 8                8     {'C': 100, 'gamma': 1}         0.946302


### tuning p by random search + k-fold ###
def fine_tune_random_search_and_k_fold():
    svm = SVC(kernel='rbf')
    params = {"C":stats.expon(scale=1), 
              "gamma":stats.expon(scale=0.01)}
    clf = RandomizedSearchCV(svm, params, cv=5,
                             return_train_score=False, n_iter=30)
    
    t1 = time.time()
    clf.fit(iris.data, iris.target) # train
    t2 = time.time()

    print("finished: {:.2f}sec".format(t2 - t1))
    result_df = pd.DataFrame(clf.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    print(result_df[["rank_test_score", 
                     "param_C",
                     "param_gamma",
                     "mean_test_score"]])
    
    ### results:
    # finished: 0.28sec
    # rank_test_score   param_C param_gamma  mean_test_score
    # 10                1  1.039456    0.051137         0.973333 <- overfeat fine_tune_grid_search_and_k_fold()!
    # 27                2  2.001711    0.026791         0.966667
    # 24                2  1.436723    0.046017         0.966667
    # 9                 2  5.128318    0.010575         0.966667
    # 19                5  0.792434    0.019051         0.953333
    # 25                6  0.808348    0.017724         0.946667
    # 18                7  2.515376    0.004497         0.933333
    # 6                 7  2.237805    0.004491         0.933333
    # 12                9   0.08947     0.03381         0.920000
    # 8                 9  2.256695    0.003684         0.920000
    # 26                9  0.745651    0.005439         0.920000
    # 15                9  1.746147    0.002507         0.920000
    # 22               13  0.003806    0.007419         0.913333
    # 21               13   0.36273    0.006676         0.913333
    # 23               13   0.50984    0.000729         0.913333
    # 20               13   0.36186    0.002981         0.913333
    # 0                13  1.249614    0.000661         0.913333
    # 14               13  0.398725    0.003374         0.913333
    # 16               13  0.233116    0.003215         0.913333
    # 28               13   1.00173    0.001605         0.913333
    # 11               13  0.075473    0.000229         0.913333
    # 7                13  0.086446    0.006448         0.913333
    # 5                13  1.770637    0.002698         0.913333
    # 4                13  0.866544    0.000033         0.913333
    # 3                13  0.566787    0.005926         0.913333
    # 2                13  0.406264    0.006313         0.913333
    # 1                13  0.483279    0.005074         0.913333
    # 17               13  0.020759    0.006633         0.913333
    # 13               29  0.809649    0.008446         0.906667
    # 29               30  4.741237    0.001304         0.900000

if __name__ == "__main__":
    svm_with_no_fine_tune_and_split_direct()
    svm_with_no_fine_tune_and_split_random()
    fine_tune_grid_search_and_k_fold()
    fine_tune_random_search_and_k_fold()
