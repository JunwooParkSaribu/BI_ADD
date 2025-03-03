import numpy as np
import pandas as pd
import os

def load_andi_datas(datapath, prefix, exp):
    fovs = np.arange(30)
    dfs = []
    vips = []
    for fov in fovs:
        if int(prefix) == 2:
            df = pd.read_csv(datapath + f'track_{prefix}/exp_{exp}/trajs_fov_{fov}.csv')
            vips.append(-1)
        else:
            df = pd.read_csv(datapath + f'track_{prefix}/exp_{exp}/videos_fov_{fov}_track.csv')
            vip_indices = np.load(f'{datapath}/track_{prefix}/exp_{exp}/videos_fov_{fov}_vip_indices.npz')[
                'andi2_indices']
            vips.append(vip_indices)
        dfs.append(df)
    return dfs, fovs, vips


def load_datas(datapath):
    dfs = []
    file_list = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df = pd.read_csv(f'{datapath}/{file}')
            dfs.append(df)
            file_list.append(file.split('.csv')[0])
    return dfs, file_list
