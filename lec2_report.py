### 検証環境: 
### macOS Montray ver.12.2.1
### python 3.11.3
### poetry 1.4.2

### Install: 
### $ poetry install
### $ poetry run python lec2_report2.py

import pandas as pd
import time

from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits

### tuning p by grid search + k-fold ###
def fine_tune_grid_search_and_k_fold():
    digits = load_digits()
    svm = SVC()
    # svm = SVC(kernel='rbf')
    # kf = KFold(n_splits=4, shuffle=True, random_state=42)
    params = {"C":[0.1, 1, 10], "gamma":[0.001, 0.01, 0.1]}
    clf = GridSearchCV(svm, params, cv=5, iid=True,
                       return_train_score=False)
    t1 = time.time()
    clf.fit(digits.data, digits.target)
    t2 = time.time()
    print("{:.2f} 秒かかった".format(t2 - t1))
    result_df = pd.DataFrame(clf.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    print(result_df[["rank_test_score", 
                     "params", 
                     "mean_test_score"]])

### tuning p by random search + k-fold ###

def fine_tune_random_search_and_k_fold():
    digits = load_digits()
    svm = SVC()
    params = {"C":stats.expon(scale=1), 
              "gamma":stats.expon(scale=0.01)}
    clf = RandomizedSearchCV(svm, params, cv=5, iid=True,
                             return_train_score=False, n_iter=30)
    t1 = time.time()
    clf.fit(digits.data, digits.target)
    t2 = time.time()
    print("{:.2f}秒かかった".format(t2 - t1))
    result_df = pd.DataFrame(clf.cv_results_)
    result_df.sort_values(
        by="rank_test_score", inplace=True)
    print(result_df[["rank_test_score", 
                     "param_C",
                     "param_gamma",
                     "mean_test_score"]])

if __name__ == "__main__":
    fine_tune_grid_search_and_k_fold()
    #fine_tune_random_search_and_k_fold
