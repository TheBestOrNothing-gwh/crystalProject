import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import numpy as np
import json

from crystalproject.visualize.drawer import draw_compare_heatmap


# Settings
alpha = 0.1
gamma = 0.1
kernel = "rbf"
test_size = 0.2
seed = 123

# descriptor
descriptor_index = {
        "Geo": [
                "density [kg\/m^3]",
                "largest included sphere diameter [A]",
                "largest free sphere diameter [A]",
                "largest included sphere along free sphere path diameter [A]",
                "void fraction [widom]",
                
        ],
        "Chem(aver)": [],
        "Chem(sum)": [],
        "Geo & Chem(aver)": [],
        "Geo & Chem(sum)": [],
}
# target
target_index = ["absolute methane uptake low P [v STP\/v]", "absolute methane uptake high P [v STP\/v]"]

# read in data
dataPath = "/home/gwh/project/crystalProject/DATA/cofs_Methane/process/id_prop_all.json"
with open(dataPath) as f:
        datas = json.load(f)
datas = pd.json_normalize(datas)
print(len(datas))
train_set, test_set = train_test_split(
        datas, test_size=test_size, shuffle=True, random_state=seed
)

for 
X_train = np.concatenate([np.array(train_set.loc[:, index].tolist()) for index in descriptor_index], axis=1)
X_test = np.concatenate([np.array(test_set.loc[:, index].tolist()) for index in descriptor_index], axis=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = np.concatenate([np.array(train_set.loc[:, index].tolist()) for index in target_index], axis=1)
y_test = np.concatenate([np.array(test_set.loc[:, index].tolist()) for index in target_index], axis=1)

krr = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
krr.fit(X_train, y_train)
y_train_pred = krr.predict(X_train)
y_test_pred = krr.predict(X_test)

# 可视化展示
jointplot = draw_compare_heatmap(y_test, y_test_pred, )