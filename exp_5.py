import pandas as pd
import numpy as np
from mlens.ensemble import SuperLearner
from sklearn import model_selection
from self_paced_ensemble import SelfPacedEnsembleClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
import lightgbm as lgb_model
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import datasets
from sklearn.preprocessing import label_binarize, PolynomialFeatures
from sklearn.multiclass import OneVsRestClassifier
from self_paced_ensemble import SelfPacedEnsembleClassifier
import matplotlib.pyplot as plt

def get_models():
    DecisionTree = DecisionTreeClassifier(random_state=SEED)  # 决策树
    RandomForest = RandomForestClassifier(n_estimators=100, max_features=3, random_state=SEED)  # 随机森林
    ExtraTrees = ExtraTreesClassifier(n_estimators=100, random_state=SEED)  # etree
    GBDT = GradientBoostingClassifier(n_estimators=100, random_state=SEED)#GDBT
    LightGBM = lgb_model.sklearn.LGBMClassifier(is_unbalance=False, learning_rate=0.04, n_estimators=110, max_bin=400,
                                           scale_pos_weight=0.8)
    adaboost=AdaBoostClassifier(random_state=SEED)#adaboost

    models = {
        'primary classifier 1': DecisionTree,
        'primary classifier 2': RandomForest,
        'primary classifier 3': ExtraTrees,
        'primary classifier 4': GBDT,
        # '5': LightGBM,
        'primary classifier 4': adaboost}
    return models

def train_predict(model_list):#预测
#将每个模型的预测值保留在DataFrame中，行是每个样本预测值，列是模型
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    cm = [plt.cm.rainbow(i)
    for i in np.linspace(0, 1.0, 20)]
    cols = list()
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(x_train, y_train)
        P.iloc[:, i] = m.predict(x_test)
        e_sl = m.predict_proba(x_test)
        y_pred = e_sl[:, 1]
        y_pred = np.around(y_pred, 0).astype(int)
        cols.append(name)
        print("done")
        # fpr, tpr, _ = roc_curve(y_test, y_pred)
        # plt.plot(fpr, tpr, label=name, c=cm[i+1],linestyle='--')
    P.columns = cols
    print("ALL model Done.\n")
    return P

def train_predict_Self(model_list):#预测
#将每个模型的预测值保留在DataFrame中，行是每个样本预测值，列是模型
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    cm = [plt.cm.rainbow(i)
    for i in np.linspace(0, 1.0, 20)]
    cols = list()

    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m = SelfPacedEnsembleClassifier(
        base_estimator = m,
        n_estimators=10)
        m.fit(x_train, y_train)
        P.iloc[:, i] = m.predict(x_test)
        e_sl = m.predict_proba(x_test)
        y_pred = e_sl[:, 1]
        y_pred = np.around(y_pred, 0).astype(int)
        cols.append(name)
        print("done")
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=name, c=cm[i + 1], linestyle='--')
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

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--')
cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, 20)]

Ecosystem_level = ["host_is_GitHub","have_key","have_readme","have_home_page","license_present","versions_present","More_than_6_months",
                             "More_than_20_months","one_point_oh","is_fork","stargazers_count","subscribers_count","forks_count","contributions_count",
                             "commit_count","any_outdated_dependencies","class","Importance of direct technology","Importance of indirect technology",
                             "Social importance","Information transmission capability","Technical independence","Social independence"]


data = pd.read_csv("./data/newmodel.csv")
Y = data.Survival
# 选择特征变量

X = data[Ecosystem_level]
X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1

SEED = 123  # 设立随机种子以便结果复现
np.random.seed(SEED)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)  # 划分数据集
rus = RandomUnderSampler(random_state=SEED)
X_undersampled, y_undersampled = rus.fit_resample(X,Y)
x_train, x_test, y_train, y_test = train_test_split(X_undersampled, y_undersampled, test_size=0.3, random_state=SEED)

baseline = DecisionTreeClassifier(random_state=SEED)
baseline.fit(x_train, y_train)
e_sl = baseline.predict_proba(x_test)
y_pred = e_sl[:, 1]
y_pred = np.around(y_pred, 0).astype(int)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
AUC = roc_auc_score(y_test, e_sl[:, 1])
print("baseline", recall, precision, accuracy, f1, AUC)
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label="baseline", c='black',linestyle=':')

# 一级分类器
print("训练一级分类器")
data = pd.read_csv("./data/newmodel.csv")
Y = data.Survival
# 选择特征变量
X = data[Ecosystem_level]
X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)

base_learners = get_models()
P = train_predict_Self(base_learners)
score_models(P, y_test)

Project_level = P.mean(axis=1)
print("一级: %.3f" % roc_auc_score(y_test, P.mean(axis=1)))

# SMOTE
print("集成")
smote = SMOTE(random_state=SEED)
X_smotesampled, y_smotesampled = smote.fit_resample(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X_smotesampled, y_smotesampled, test_size=0.2, random_state=SEED)  # 划分数据集

base_learners = get_models()
P = train_predict(base_learners)
score_models(P, y_test)

Project_level = P.mean(axis=1)
print("集成: %.3f" % roc_auc_score(y_test, P.mean(axis=1)))


HSPM = P.mean(axis=1)
HSPM = np.around(HSPM, 0).astype(int)
recall = recall_score(y_test, HSPM)
precision = precision_score(y_test, HSPM)
accuracy = accuracy_score(y_test, HSPM)
f1 = f1_score(y_test, HSPM)
AUC = roc_auc_score(y_test, P.mean(axis=1))
print("HSPM", recall, precision, accuracy, f1, AUC)
fpr, tpr, _ = roc_curve(y_test, HSPM)
plt.plot(fpr, tpr, label="HSPM", c='red',linestyle='-')
print("开始集成学习")
# 训练二级分类器:集成学习
meta_learner = RandomForestClassifier(n_estimators=100, max_features=3, random_state=SEED)

meta_learner.fit(P, y_test)  # 用一级学习器的预测值作为元学习器的输入，并拟合元学习器，元学习器一定要拟合，不然无法集成。

sl = SuperLearner(folds=3, random_state=SEED, verbose=2, backend="multiprocessing")

sl.add(list(base_learners.values()), proba=True)  # 加入基学习器
sl.add_meta(meta_learner, proba=True)  # 加入元学习器
# 训练集成模型

sl.fit(x_train, y_train)
# 预测
e_sl = sl.predict_proba(x_test)
y_pred = e_sl[:, 1]
y_pred = np.around(y_pred, 0).astype(int)

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
AUC = roc_auc_score(y_test, e_sl[:, 1])
print("EL-HSPM", recall, precision, accuracy, f1, AUC)

# 绘制二级分类器ROC曲线
fpr, tpr, _ = roc_curve(y_test, e_sl[:, 1])
plt.plot(fpr, tpr, label="EL-HSPM", c='red',linestyle='-')


# 输出图片
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right", frameon=False)
plt.show()


