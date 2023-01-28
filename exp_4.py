import pandas as pd
import numpy as np
from mlens.visualization import corrmat
from self_paced_ensemble import SelfPacedEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from self_paced_ensemble import SelfPacedEnsembleClassifier
def get_models():
    DecisionTree = DecisionTreeClassifier(random_state=SEED)  # 决策树
    RandomForest = RandomForestClassifier(n_estimators=100, max_features=3, random_state=SEED)  # 随机森林
    ExtraTrees = ExtraTreesClassifier(n_estimators=100, random_state=SEED)  # etree
    GBDT = GradientBoostingClassifier(n_estimators=100, random_state=SEED)#GDBT
    LightGBM = lgb_model.sklearn.LGBMClassifier(is_unbalance=False, learning_rate=0.04, n_estimators=110, max_bin=400,
                                           scale_pos_weight=0.8)
    adaboost=AdaBoostClassifier(random_state=SEED)#adaboost

    models = {
        '1': DecisionTree,
        '2': RandomForest,
        '3': ExtraTrees,
        '4': GBDT,
        # '5': LightGBM,
        '5': adaboost}
    return models

def train_predict(model_list):#预测
#将每个模型的预测值保留在DataFrame中，行是每个样本预测值，列是模型
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m = SelfPacedEnsembleClassifier(
            base_estimator=m,
            n_estimators=10,
        )
        m.fit(x_train, y_train)
        P.iloc[:, i] = m.predict(x_test)
        cols.append(name)
        print("done")
    P.columns = cols
    print("ALL model Done.\n")
    return P

def score_models(P, y):#打印评估值
    print("模型评估")
    for m in P.columns:
        recall = recall_score(y, P.loc[:, m])
        precision = precision_score(y, P.loc[:, m])
        accuracy = accuracy_score(y, P.loc[:, m])
        f1 = f1_score(y, P.loc[:, m])
        AUC = roc_auc_score(y, P.loc[:, m])

        print(m,recall, precision,accuracy,f1,AUC)

    print("Done.\n")


Ecosystem_level = ["host_is_GitHub","have_key","have_readme","have_home_page","license_present","versions_present","More_than_6_months",
                             "More_than_20_months","one_point_oh","is_fork","stargazers_count","subscribers_count","forks_count","contributions_count",
                             "commit_count","any_outdated_dependencies","class","Importance of direct technology","Importance of indirect technology",
                             "Social importance","Information transmission capability","Technical independence","Social independence"]

#
# data = pd.read_csv("./data/newmodel.csv")
#
# # 选择预测目标变量
# Y = data.Survival
# # 选择特征变量
# X = data[Ecosystem_level]
# X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1
# SEED = 123  # 设立随机种子以便结果复现
# SEED = np.random.seed(SEED)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
# base_learners = get_models()
# P1 = train_predict(base_learners)
# score_models(P1, y_test)
# corrmat(P1.corr(), inflate=False, center=0.5)
# print("结束")

SEED = 123  # 设立随机种子以便结果复现
SEED = np.random.seed(SEED)

data = pd.read_csv("./data/newmodel.csv")
data = data.sample(n = 100000, random_state = SEED)
# 选择预测目标变量
Y = data.Survival
# 选择特征变量
X = data[Ecosystem_level]
X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)
base_learners = get_models()
P2 = train_predict(base_learners)
score_models(P2, y_test)
corrmat(P2.corr(), inflate=False, center=0.5)
print("结束")