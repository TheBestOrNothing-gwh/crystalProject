import os
import pandas as pd
from tqdm import tqdm
import shutil
import random
import pickle
import multiprocessing
from crystalproject.data.prepare.process import *


def err_call_back(err):
    print(f'出错啦~ error:{str(err)}')


def func_simple(root_dir, target_dir, id):
    shutil.copy(
        os.path.join(root_dir, id+".cif"),
        os.path.join(target_dir, id+".cif")
    )

def func_cg(root_dir, target_dir, id, radius=8, max_nbr_num=12):
    process_data = create_crystal_graph(
        os.path.join(root_dir, id+".cif"),
        radius,
        max_nbr_num,
    )
    f_save = open(
        os.path.join(target_dir, id+"_radius.pkl"),
        "wb"
    )
    pickle.dump(process_data, f_save)
    f_save.close()

def func_topo(root_dir, target_dir, row, radius=8, max_nbr_num=12):
    process_data = create_crystal_topo(
        os.path.join(root_dir, row["name"]+".cif"),
        radius,
        max_nbr_num,
        row["use_bond_types"],
        row["bond_types"],
        row["linker_types"],
    )
    f_save = open(
        os.path.join(target_dir, id+"_topo.pkl"),
        "wb"
    )
    pickle.dump(process_data, f_save)
    f_save.close()


def pre_control(root_dir, target_dir, datas, stage="crystalGraph", radius=8, max_nbr_num=12, processes=24):
    pool = multiprocessing.Pool(processes=processes)
    # pbar = tqdm(total=len(datas))
    # pbar.set_description("process data")
    # update = lambda *args: pbar.update()
    match stage:
        case "simple":
            for _, row in datas.iterrows():
                pool.apply_async(
                    func_simple,
                    (root_dir, target_dir, row["name"]),
                    callback=update,
                    error_callback=err_call_back
                )
        case "crystalGraph":
            for _, row in datas.iterrows():
                pool.apply_async(
                    func_cg,
                    (root_dir, target_dir, row["name"], radius, max_nbr_num),
                    callback=update,
                    error_callback=err_call_back
                )
        case "crystalTopo":
            for _, row in datas.iterrows():
                # pool.apply_async(
                #     func_topo,
                #     (root_dir, target_dir, row, radius, max_nbr_num),
                #     callback=update,
                #     error_callback=err_call_back
                # )
                print(row["name"])
                func_topo(root_dir, target_dir, row, radius, max_nbr_num)
        case _:
            print("No such data format.")
    pool.close()
    pool.join()
    datas.to_csv(os.path.join(target_dir, "id_prop.csv"), index=0)


def prepare_data(root_dir, target_dir, split=[], stage="simple", radius=8, processes=24):
    datas = pd.read_csv(os.path.join(root_dir, "id_prop.csv"))
    if len(split) != 0:
        assert (
            abs(split[0] + split[1] + split[2] - 1) <= 1e-5
        ), "train + val + test == 1"
        # random.seed(2023)
        # 设置好pandas的随机数，让每一次划分数据集都是相同的
        datas = datas.sample(frac=1.0)
        split_index_1, split_index_2 = int(datas.shape[0] * split[0]), int(datas.shape[0] * (split[0] + split[1]))
        os.makedirs(os.path.join(target_dir, "train"))
        pre_control(root_dir, os.path.join(target_dir, "train"), datas.iloc[0:split_index_1, :],
                    stage=stage, radius=radius, processes=processes)
        os.makedirs(os.path.join(target_dir, "val"))
        pre_control(root_dir, os.path.join(target_dir, "val"), datas.iloc[split_index_1:split_index_2, :],
                    stage=stage, radius=radius, processes=processes)
        os.makedirs(os.path.join(target_dir, "test"))
        pre_control(root_dir, os.path.join(target_dir, "test"), datas.iloc[split_index_2:, :],
                    stage=stage, radius=radius, processes=processes)
    else:
        pre_control(root_dir, target_dir, datas,
                    stage=stage, radius=radius, processes=processes)


if __name__ == "__main__":
    prepare_data(
        "/home/gwh/project/crystalProject/DATA/cofs_Methane/debug_structures_small",
        "/home/gwh/project/crystalProject/DATA/cofs_Methane/debug_structures_small_test"
    )
