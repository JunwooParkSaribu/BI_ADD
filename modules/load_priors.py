import os
import sys
import json
import numpy as np


def load_state_prior(savepath):
    json_path = f'{savepath}/priors/nb_states_prior.json'
    with open(json_path) as f:
        states_json = json.load(f)
        states_prior = states_json[f'{savepath}']
        f.close()
    return states_prior


def load_priors(path):
    try:
        loaded = np.load(f'{path}/priors/priors_data.npz')
        all_alphas = loaded['alphas']
        all_seg_lengths = loaded['seg_lengths']
        all_ks = loaded['all_ks'].flatten()
        return all_alphas, all_ks, all_seg_lengths
    except Exception as e:
        sys.exit(f'Can\'t load priors of data')


def build_state_priror(savepath, input_files, init_nb_state=2):
    json_path = f'{savepath}/priors/nb_states_prior.json'
    if not os.path.exists(f'{savepath}/priors'):
        os.makedirs(f'{savepath}/priors')

    if not os.path.exists(json_path):
        init_states = {f'{savepath}': init_nb_state}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(init_states, f, ensure_ascii=False, indent=4)
            f.close()
