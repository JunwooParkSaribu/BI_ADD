import os
import sys
import json
import numpy as np


def load_state_prior(savepath, prefix, exp):
    json_path = f'{savepath}/priors/nb_states_prior.json'
    with open(json_path) as f:
        states_json = json.load(f)
        states_prior = states_json[str(prefix)][str(exp)]
        f.close()
    return states_prior


def load_priors(path, prefix, exp):
    try:
        loaded = np.load(f'{path}/priors/priors_{prefix}_{exp}.npz')
        all_alphas = loaded['alphas']
        all_seg_lengths = loaded['seg_lengths']
        all_ks = loaded['all_ks'].flatten()
        return all_alphas, all_ks, all_seg_lengths
    except Exception as e:
        sys.exit(f'Can\'t load priors of prefix:{prefix}, exp:{exp}')


def build_state_priror(savepath, prefixs, exp_list):
    json_path = f'{savepath}/priors/nb_states_prior.json'
    if not os.path.exists(f'{savepath}/priors'):
        os.makedirs(f'{savepath}/priors')

    if not os.path.exists(json_path):
        init_states = {}
        for prefix in prefixs:
            init_states[prefix] = {int(exp): -1 for exp in exp_list}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(init_states, f, ensure_ascii=False, indent=4)
            f.close()
