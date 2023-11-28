import os
import pandas as pd
from tqdm import tqdm
import shutil
import pickle
import multiprocessing
import json

from crystalproject.data.prepare.process.crystal_topo import create_crystal_topo


def err_call_back(err):
    print(f'出错啦~ error:{str(err)}')


def func_simple(root_dir, target_dir, id):
    shutil.copy(
            os.path.join(root_dir, id+".cif"),
            os.path.join(target_dir, id+".cif")
        )

def func_topo(root_dir, target_dir, row, radius=5.0, max_nbr_num=12):
    if os.path.exists(os.path.join(target_dir, row["name"]+".pkl")):
        return 
    process_data = create_crystal_topo(
        os.path.join(root_dir, row["name"]+".cif"),
        radius,
        max_nbr_num,
        row["use_bond_types"],
        row["bond_types"],
        row["linker_types"],
    )
    f_save = open(
        os.path.join(target_dir, row["name"]+".pkl"),
        "wb"
    )
    pickle.dump(process_data, f_save)
    f_save.close()

def func_RACs():
    TODO

def pre_control(root_dir, target_dir, datas, stage="crystalTopo", radius=5.0, max_nbr_num=12, processes=24):
    pool = multiprocessing.Pool(processes=processes)
    pbar = tqdm(total=len(datas))
    pbar.set_description("process data")
    update = lambda *args: pbar.update()
    match stage:
        case "simple":
            for _, row in datas.iterrows():
                pool.apply_async(
                    func_simple,
                    (root_dir, target_dir, row["name"]),
                    callback=update,
                    error_callback=err_call_back
                )
        case "crystalTopo":
            for _, row in datas.iterrows():
                pool.apply_async(
                    func_topo,
                    (root_dir, target_dir, row, radius, max_nbr_num),
                    callback=update,
                    error_callback=err_call_back
                )
        case _:
            print("No such data format.")
    pool.close()
    pool.join()
    files = os.listdir(target_dir)
    files = [file.split(".")[0] for file in files]
    datas = datas[datas["name"].isin(files)]
    return datas


def prepare_data(root_dir, target_dir, split=[0.8, 0.1, 0.1], stage="simple", radius=5.0, processes=24):
    with open(os.path.join(root_dir, "id_prop.json")) as f:
        datas = json.load(f)
    datas = pd.json_normalize(datas)
    # 进行数据处理，并返回处理完后所有剩余的部分
    if not os.path.exists(os.path.join(target_dir, "all")):
        os.makedirs(os.path.join(target_dir, "all"))
    datas = pre_control(root_dir, os.path.join(target_dir, "all"), datas.iloc[:, :],
                stage=stage, radius=radius, processes=processes)
    datas.to_json(os.path.join(target_dir, "id_prop_all.json"), orient="records", force_ascii=True, indent=4)
    if len(split) != 0:
        assert (
            abs(split[0] + split[1] + split[2] - 1) <= 1e-5
        ), "train + val + test == 1"
        # random.seed(2023)
        # 设置好pandas的随机数，让每一次划分数据集都是相同的
        datas = datas.sample(frac=1.0)
        # 划分数据集
        split_index_1, split_index_2 = int(datas.shape[0] * split[0]), int(datas.shape[0] * (split[0] + split[1]))
        train_datas = datas.iloc[:split_index_1, :]
        val_datas = datas.iloc[split_index_1:split_index_2, :]
        test_datas = datas.iloc[split_index_2:, :]
        train_datas.to_json(os.path.join(target_dir, "id_prop_train.json"), orient="records", force_ascii=True, indent=4)
        val_datas.to_json(os.path.join(target_dir, "id_prop_val.json"), orient="records", force_ascii=True, indent=4)
        test_datas.to_json(os.path.join(target_dir, "id_prop_test.json"), orient="records", force_ascii=True, indent=4)


if __name__ == "__main__":
    prepare_data(
        "/home/bachelor/gwh/project/crystalProject/DATA/cofs_Methane/structures_primitive",
        "/home/bachelor/gwh/project/crystalProject/DATA/cofs_Methane/structures_primitive_process",
        stage="crystalTopo"
    )
