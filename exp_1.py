import pandas as pd
import numpy as np
import csv
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from mlens.ensemble import SuperLearner
from mlens.visualization import corrmat
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def writeAUC(num):
    newfile = "./AUC-train" + str(num) + ".csv"
    writeFile = open(newfile, "w+", newline='')
    writer = csv.writer(writeFile)
    return writer


# 基学习器的训练
def get_models():
    svc = SVC(C=1, random_state=SEED, kernel="linear", probability=True)  #用于分类的支持向量机
    k_NN = KNeighborsClassifier(n_neighbors=3)  # K近邻聚类
    Logistic = LogisticRegression(C=100, solver='liblinear', random_state=SEED)  # 逻辑回归
    NBC = GaussianNB()#朴素贝叶斯
    MLP = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)#多层感知器
    DecisionTree = DecisionTreeClassifier(random_state=SEED)  # 决策树
    RandomForest = RandomForestClassifier(n_estimators=100, max_features=3, random_state=SEED)  # 随机森林
    ExtraTrees = ExtraTreesClassifier(n_estimators=100, random_state=SEED)  # etree
    GBDT = GradientBoostingClassifier(n_estimators=100, random_state=SEED)#GDBT
    LightGBM = lgb_model.sklearn.LGBMClassifier(is_unbalance=False, learning_rate=0.04, n_estimators=110, max_bin=400,
                                           scale_pos_weight=0.8)
    adaboost=AdaBoostClassifier(random_state=SEED)#adaboost

    models = {
        'k-NN': k_NN,
        # 'Logistic': Logistic,
        # 'NBC': NBC,
        # 'MLP': MLP,
        'DecisionTree': DecisionTree,
        'RandomForest': RandomForest,
        'ExtraTrees': ExtraTrees,
        'GBDT': GBDT,
        'LightGBM': LightGBM,
        'adaboost': adaboost}
    return models

def train_predict(model_list):#预测
#将每个模型的预测值保留在DataFrame中，行是每个样本预测值，列是模型
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
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
        confusion = confusion_matrix(y, P.loc[:, m],labels=[0,1])
        f1 = f1_score(y, P.loc[:, m])
        AUC = roc_auc_score(y, P.loc[:, m])

        print(m,recall, precision,accuracy,f1,AUC)
        # print(confusion)
        # newitem = [m,recall,precision,f1,AUC]
        # writeAUC(i+1).writerow(newitem)
    print("Done.\n")

def plot_roc_curve_PL(ytest, P_base_learners, labels):
    cm = [plt.cm.rainbow(i)
    for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1],linestyle=':')

def plot_roc_curve_EL(ytest, P_base_learners, labels):
    cm = [plt.cm.rainbow(i)
    for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1],linestyle='-')



Project_level = ["host_is_GitHub","have_key","have_readme","have_home_page","license_present","versions_present","More_than_6_months",
                             "More_than_20_months","one_point_oh","is_fork","stargazers_count","subscribers_count","forks_count","contributions_count",
                             "commit_count"]

Ecosystem_level = ["host_is_GitHub","have_key","have_readme","have_home_page","license_present","versions_present","More_than_6_months",
                             "More_than_20_months","one_point_oh","is_fork","stargazers_count","subscribers_count","forks_count","contributions_count",
                             "commit_count","any_outdated_dependencies","class","Importance of direct technology","Importance of indirect technology",
                             "Social importance","Information transmission capability","Technical independence","Social independence"]


if __name__ == "__main__":
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    data = pd.read_csv("./data/newmodel.csv")

    # Project_level
    # 选择预测目标变量
    Y = data.Survival
    # 选择特征变量

    X = data[Project_level]
    X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1

    SEED = 123  # 设立随机种子以便结果复现
    np.random.seed(SEED)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)  # 划分数据集

    # 训练
    base_learners = get_models()
    P = train_predict(base_learners)
    score_models(P, y_test)
    plot_roc_curve_PL(y_test, P.values, list(P.columns))
    # Project_level = P.mean(axis=1)
    # print("Project_level: %.3f" % roc_auc_score(y_test, P.mean(axis=1)))


    # Ecosystem_level
    X = data[Ecosystem_level]
    X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1
    SEED = 123  # 设立随机种子以便结果复现
    np.random.seed(SEED)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)  # 划分数据集

    # 训练
    base_learners = get_models()
    P = train_predict(base_learners)
    score_models(P, y_test)
    plot_roc_curve_EL(y_test, P.values, list(P.columns))
    # Ecosystem_level = P.mean(axis=1)
    # print("Ecosystem_level: %.3f" % roc_auc_score(y_test, P.mean(axis=1)))
    # fpr, tpr, _ = roc_curve(y_test, Ecosystem_level)
    # plt.plot(fpr, tpr, label="Ecosystem_level model", c=cm[6])

    # 输出图片
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right", frameon=False)
    plt.show()
