import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from ana_prediction import exhaustive_cps_search
from load_files import load_andi_files
from load_models import RegModel
from visualization import raw_distribution


def make_priors(datapath, savepath, imagepath, make_image):
    reg_model_nums = [5, 8, 12, 16, 32, 64, 128]
    ext_width = 100
    win_widths = np.arange(20, 40, 2)
    shift_width = 5
    if not os.path.exists(f'{savepath}/priors/priors_data.npz'):
        reg_model = RegModel(reg_model_nums)
        all_seg_lengths = []
        all_alphas = []
        all_ks = []
        dfs, _ = load_andi_files.load_datas(datapath)
        for df in dfs:
            traj_idx = np.sort(df.traj_idx.unique())
            for idx in traj_idx:
                x = np.array(df[df.traj_idx == idx])[:, 2]
                y = np.array(df[df.traj_idx == idx])[:, 3]
                if len(x) > 5:
                    cps, alphas, ks, state, seg_lengths = exhaustive_cps_search(x, y,
                                                                                win_widths,
                                                                                shift_width,
                                                                                ext_width,
                                                                                search_seuil=0.10,
                                                                                cluster=None,
                                                                                cluster_states=None,
                                                                                reg_model=reg_model,
                                                                                )
                    all_alphas.extend(alphas)
                    all_seg_lengths.extend(seg_lengths)
                    all_ks.extend(ks)

        all_alphas = np.array(all_alphas)
        all_seg_lengths = np.array(all_seg_lengths)
        all_ks = np.array(all_ks)
        np.savez(f'{savepath}/priors/priors_data.npz',
                 alphas=all_alphas, seg_lengths=all_seg_lengths, all_ks=all_ks)

        if make_image:
            raw_distribution(f'{imagepath}/sample_distribs.png', all_alphas, all_ks, all_seg_lengths)
            print('Sample distribution generated...')


if __name__ == "__main__":
    make_priors(datapath=sys.argv[1], savepath=sys.argv[2], imagepath=sys.argv[3], make_image=bool(sys.argv[4]))
