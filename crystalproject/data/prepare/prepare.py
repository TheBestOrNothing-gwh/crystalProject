import os
import json
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

def func_cg(root_dir, target_dir, id, radius):
    process_data = create_crystal_graph(
        os.path.join(root_dir, id+".cif"),
        radius
    )
    f_save = open(
        os.path.join(target_dir, id+".pkl"),
        "wb"
    )
    pickle.dump(process_data, f_save)
    f_save.close()

def func_topo(root_dir, target_dir, id):
    process_data = create_crystal_topo(
        os.path.join(root_dir, id+".cif")
    )
    f_save = open(
        os.path.join(target_dir, id+".pkl"),
        "wb"
    )
    pickle.dump(process_data, f_save)
    f_save.close()


def pre_control(root_dir, target_dir, id_props, stage="crystalGraph", radius=8, processes=24):
    pool = multiprocessing.Pool(processes=processes)
    ids = list(id_props.keys())
    pbar = tqdm(total=len(ids))
    pbar.set_description("process data")
    update = lambda *args: pbar.update()
    match stage:
        case "simple":
            for id in ids:
                pool.apply_async(
                    func_simple,
                    (root_dir, target_dir, id),
                    callback=update,
                    error_callback=err_call_back
                )
        case "crystalGraph":
            for id in ids:
                pool.apply_async(
                    func_cg,
                    (root_dir, target_dir, id, radius),
                    callback=update,
                    error_callback=err_call_back
                )
        case "crystalTopo":
            for id in ids:
                pool.apply_async(
                    func_topo,
                    (root_dir, target_dir, id),
                    callback=update,
                    error_callback=err_call_back
                )
        case _:
            print("No such data format.")
    pool.close()
    pool.join()
    with open(os.path.join(target_dir, "id_prop.json"), "w") as f:
        f.write(json.dumps(id_props, indent=4, ensure_ascii=False))


def prepare_data(root_dir, target_dir, split=[], stage="simple", radius=8, processes=24):
    with open(os.path.join(root_dir, "id_prop.json"), "r") as f:
        id_props = json.load(f)
        ids = list(id_props.keys())
    if len(split) != 0:
        assert (
            abs(split[0] + split[1] + split[2] - 1) <= 1e-5
        ), "train + val + test == 1"
        random.seed(2023)
        random.shuffle(ids)
        train_id_props, val_id_props, test_id_props = {}, {}, {}
        for i, id in enumerate(ids):
            if i < int(len(ids) * split[0]):
                train_id_props[id] = id_props[id]
            elif i < int(len(ids) * (split[0] + split[1])):
                val_id_props[id] = id_props[id]
            else:
                test_id_props[id] = id_props[id]
        os.makedirs(os.path.join(target_dir, "train"))
        pre_control(root_dir, os.path.join(target_dir, "train"), train_id_props,
                    stage=stage, radius=radius, processes=processes)
        os.makedirs(os.path.join(target_dir, "val"))
        pre_control(root_dir, os.path.join(target_dir, "val"), val_id_props,
                    stage=stage, radius=radius, processes=processes)
        os.makedirs(os.path.join(target_dir, "test"))
        pre_control(root_dir, os.path.join(target_dir, "test"), test_id_props,
                    stage=stage, radius=radius, processes=processes)
    else:
        pre_control(root_dir, target_dir, id_props,
                    stage=stage, radius=radius, processes=processes)


if __name__ == "__main__":
    prepare_data(
        "/home/gwh/project/crystalProject/DATA/cofs_Methane/debug_structures_small",
        "/home/gwh/project/crystalProject/DATA/cofs_Methane/debug_structures_small_test"
    )
