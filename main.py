import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn import metrics

city_num = {'GD': 0, 'ZJ': 1, 'SC': 2, 'HB': 3, 'HLJ': 4, 'NM': 5, 'LN': 6, 'JL': 7, 'AH': 8, 'FJ': 9, 'JX': 10,
            'SD': 11, 'HuB': 12, 'HNa': 13, 'GX': 14, 'GZ': 15, 'YN': 16, 'SX': 17, 'GS': 18, 'QH': 19, 'HaN': 20,
            'HN': 21, 'SXb': 22, 'TW': 23, 'NX': 24, 'XJ': 25, 'XZ': 26, 'BJ': 27, 'TJ': 28, 'SH': 29, 'CQ': 30,
            'JS': 31
            }


def to_num(city):
    if city in city_num:
        return city_num[city]
    else:
        return 32


if __name__ == '__main__':
    info = pd.read_csv('data/info4.csv')
    info['result'].fillna(0, inplace=True)
    info.dropna(inplace=True)
    info.to_csv('data/info4.csv')
    x = info[['first_order_price', 'age_month', 'city_num', 'platform_num', 'login_day', 'login_diff_time',
              'distance_day', 'login_time', 'launch_time', 'camp_num', 'learn_num', 'finish_num', 'study_num', 'coupon',
              'course_order_num', 'main_home', 'main_mime', 'coupon_visit', 'baby_info', 'share', 'click_dialog',
              'subscribe_num', 'place_count', 'next_count', 'chapter_course_count', 'year'
              ]]
    y = info['result']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    joblib.dump(gc, 'model.m')
    y_pre = gc.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_pre)
    mae = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("MSE: %.4f" % mse)
    print("MAE: %.4f" % mae)
    print("R2: %.4f" % R2)
    print("随机森林预测的准确率为：", gc.score(x_test, y_test))
    print("在交叉验证当中验证的最好结果：", gc.best_score_)
    print("gc选择了的模型K值是：", gc.best_estimator_)
    print("每次交叉验证的结果为：", gc.cv_results_)
    # Plot feature importance
    # 得到特征重要度分数
    importances_values = gc.feature_importances_
    importances = pd.DataFrame(importances_values, columns=["importance"])
    feature_data = pd.DataFrame(x_train.columns, columns=["feature"])
    importance = pd.concat([feature_data, importances], axis=1)
    # 倒叙排序
    importance = importance.sort_values(["importance"], ascending=True)
    importance["importance"] = (importance["importance"] * 1000).astype(int)
    importance = importance.sort_values(["importance"])
    importance.set_index('feature', inplace=True)
    importance.plot.barh(color='r', alpha=0.7, rot=0, figsize=(8, 8))
    plt.show()
