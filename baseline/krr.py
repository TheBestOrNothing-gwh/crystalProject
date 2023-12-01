import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import numpy as np
import json
from multiprocessing import Pool
from matplotlib import pyplot as plt
import os

from crystalproject.visualize.drawer import draw_compare


# Settings
alpha = 0.1
gamma = 0.1
kernel = "rbf"
test_size = 0.2
seed = 123

# descriptor
descriptor_indexes = {
    "Geo": ["vol", "rho", "di", "df", "dif", "asa", "av", "nasa", "nav"],
    "Chem(aver)": ["LigandRAC_mean", "FullLinkerRAC_mean", "LinkerConnectingRAC_mean", "FunctionalGroupRAC_mean"],
    "Chem(sum)": ["LigandRAC_sum", "FullLinkerRAC_sum", "LinkerConnectingRAC_sum", "FunctionalGroupRAC_sum"],
    "Geo_&_Chem(aver)": ["vol", "rho", "di", "df", "dif", "asa", "av", "nasa", "nav", "LigandRAC_mean", "FullLinkerRAC_mean", "LinkerConnectingRAC_mean", "FunctionalGroupRAC_mean"],
    "Geo_&_Chem(sum)": ["vol", "rho", "di", "df", "dif", "asa", "av", "nasa", "nav", "LigandRAC_sum", "FullLinkerRAC_sum", "LinkerConnectingRAC_sum", "FunctionalGroupRAC_sum"],
}
# target
target_indexes = {
    "CH4_lowP": ["absolute methane uptake low P [v STP/v]"], 
    "CH4_highP": ["absolute methane uptake high P [v STP/v]"], 
    "CO2_kH": ["CO2 kH [mol/kg/Pa]"], 
    "CO2_Qst": ["CO2 Qst [kJ/mol]"]
}


def fun_krr(alpha, gamma, kernel, X_train, X_test, y_train, y_test, target_index, descriptor, target):
    krr = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
    krr.fit(X_train, y_train)
    y_train_pred = krr.predict(X_train)
    y_test_pred = krr.predict(X_test)
    fig, ax = plt.subplots()
    addition = "\n".join(
        [
            f"MAE = {mean_absolute_error(y_test, y_test_pred)}",
            f"R2 = {r2_score(y_test, y_test_pred)}",
            # f"rho = {spearmanr(y_test, y_test_pred)[0]}"
        ]
    )
    draw_compare(
        fig=fig, 
        ax=ax, 
        x=y_test, 
        y=y_test_pred, 
        x_label=f"Mol.Sim: {target_index}", 
        y_label=f"ML: {target_index}", 
        addition=addition, 
        title=descriptor.replace("_", " ")
    )
    fig.savefig(
        os.path.join(os.getcwd(), "result", target+descriptor+".png"),
        bbox_inches='tight'
    )

def err_call_back(err):
    print(f'出错啦~ error:{str(err)}')


# read in data
dataPath = "/home/gwh/project/crystalProject/DATA/cofs_Methane/process/id_prop_all.json"
with open(dataPath) as f:
    datas = json.load(f)
datas = pd.json_normalize(datas)
train_set, test_set = train_test_split(
    datas,
    test_size=test_size,
    shuffle=True,
    random_state=seed
)

# pool
pool = Pool(processes=4)

for target, target_index in target_indexes.items():
    for descriptor, descriptor_index in descriptor_indexes.items():
        X_train = np.concatenate([np.array(train_set.loc[:, index].tolist()).reshape(len(train_set), -1) for index in descriptor_index], axis=1)
        X_test = np.concatenate([np.array(test_set.loc[:, index].tolist()).reshape(len(test_set), -1) for index in descriptor_index], axis=1)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = np.concatenate([np.array(train_set.loc[:, index].tolist()).reshape(len(train_set), -1) for index in target_index], axis=1)
        y_test = np.concatenate([np.array(test_set.loc[:, index].tolist()).reshape(len(test_set), -1) for index in target_index], axis=1)

        pool.apply_async(
            fun_krr,
            (alpha, gamma, kernel, X_train, X_test, y_train, y_test, target_index, descriptor, target),
            error_callback=err_call_back
        )

pool.close()
pool.join()