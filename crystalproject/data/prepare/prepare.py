import os
import pandas as pd
from tqdm import tqdm
import shutil
import pickle
import multiprocessing



def err_call_back(err):
    print(f'出错啦~ error:{str(err)}')


def func_simple(root_dir, target_dir, id):
    shutil.copy(
        os.path.join(root_dir, id+".cif"),
        os.path.join(target_dir, id+".cif")
    )

def func_topo(root_dir, target_dir, row, radius=8, max_nbr_num=12):
    from crystalproject.data.prepare.process.crystal_topo import create_crystal_topo
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


def pre_control(root_dir, target_dir, datas, stage="crystalTopo", radius=8, max_nbr_num=12, processes=24):
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
    datas.to_json(os.path.join(target_dir, "id_prop.jsonl"), orient="records", lines=True)


def prepare_data(root_dir, target_dir, split=[0.8, 0.1, 0.1], stage="simple", radius=8, processes=24):
    datas = pd.read_json(os.path.join(root_dir, "id_prop.jsonl"), orient="records", lines=True)
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
        "/home/bachelor/gwh/project/crystalProject/DATA/cofs_Methane/structures_primitive",
        "/home/bachelor/gwh/project/crystalProject/DATA/cofs_Methane/structures_primitive_process",
        stage="crystalTopo"
    )
