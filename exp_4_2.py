import pandas as pd
import numpy as np
from mlens.ensemble import SuperLearner
from mlens.visualization import corrmat
from self_paced_ensemble import SelfPacedEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score, precision_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from self_paced_ensemble import SelfPacedEnsembleClassifier
import matplotlib.pyplot as plt
#画包括集成模型的roc曲线
def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    cm = [plt.cm.rainbow(i)
    for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])
    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right",frameon=False)
    plt.show()

Ecosystem_level = ["host_is_GitHub","have_key","have_readme","have_home_page","license_present","versions_present","More_than_6_months",
                             "More_than_20_months","one_point_oh","is_fork","stargazers_count","subscribers_count","forks_count","contributions_count",
                             "commit_count","any_outdated_dependencies","class","Importance of direct technology","Importance of indirect technology",
                             "Social importance","Information transmission capability","Technical independence","Social independence"]
P = np.zeros((24000, 5))
P = pd.DataFrame(P)
cols = list()
print("Fitting models.")
for i in range(5):

    SEED = 123  # 设立随机种子以便结果复现
    np.random.seed(SEED)
    data_address = "./data/newmodel"+str(i+1)+".csv"
    data = pd.read_csv(data_address, encoding="gbk")
    data = data.sample(n = 80000, random_state = SEED)
    name = str(i+1)
    # 选择预测目标变量
    Y = data.Survival
    # 选择特征变量
    X = data[Ecosystem_level]
    X = X.add(np.ones(len(X)), axis=0)  # 增加偏置列，默认为1

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)

    m = SelfPacedEnsembleClassifier(
        base_estimator=GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        n_estimators=10,
    )
    m.fit(x_train, y_train)
    P.iloc[:, i] = m.predict(x_test)
    cols.append(name)

    e_sl = m.predict_proba(x_test)
    y_pred = e_sl[:, 1]
    y_pred = np.around(y_pred, 0).astype(int)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, e_sl[:, 1])
    print(name, recall, precision, accuracy, f1, AUC)
P.columns = cols
print("ALL model Done.\n")

corrmat(P.corr(), inflate=False)
