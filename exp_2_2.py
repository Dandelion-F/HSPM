import pandas as pd
import numpy as np
import csv

from boto import sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot


Ecosystem_level = ["host_is_GitHub","have_key","have_readme","have_home_page","license_present","versions_present","More_than_6_months",
                             "More_than_20_months","one_point_oh","is_fork","stargazers_count","subscribers_count","forks_count","contributions_count",
                             "commit_count","any_outdated_dependencies","class","Importance of direct technology","Importance of indirect technology",
                             "Social importance","Information transmission capability","Technical independence","Social independence"]

data = pd.read_csv("./data/newmodel.csv")

Y = data.Survival
X = data[Ecosystem_level]
X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1
SEED = 123  # 设立随机种子以便结果复现
np.random.seed(SEED)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)  # 划分数据集

model = XGBClassifier()
model.fit(X, Y)

plot_importance(model)
pyplot.show()
