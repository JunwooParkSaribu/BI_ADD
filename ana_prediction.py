import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from modules import load_priors
from modules.load_files.load_andi_files import load_datas
from modules.visualization import cluster_boundary, sample_distribution, cps_visualization
from modules.load_models import RegModel


REG_MODEL_NUMS = [5, 8, 12, 16, 32, 64, 128]
DEL_CONDITIONS = [1e-5, 0.001, 0.025, 0.10]
SEARCH_SEUIL = 0.15
MAX_DENSITY_NB = 25
EXT_WIDTH = 100
CLUSTER_RANGE = np.arange(1, 4)
CLUSTER_SEG_LENGTH_SEUIL = 16
ADJUST_MIN_SEG_LENGTH = 5


def gmm_bic_score(estimator, x):
    return -estimator.bic(x)


def subtraction(xs: np.ndarray):
    assert xs.ndim == 1
    uncum_list = [0.]
    for i in range(1, len(xs)):
        uncum_list.append(xs[i] - xs[i-1])
    return np.array(uncum_list)


def radius(xs: np.ndarray, ys: np.ndarray):
    rad_list = [0.]
    for i in range(1, len(xs)):
        rad_list.append(np.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2))
    return np.array(rad_list)


def make_signal(x_pos, y_pos, win_widths):
    all_vals = []
    for win_width in win_widths:
        if win_width >= len(x_pos):
            continue
        vals = []
        for checkpoint in range(win_width // 2, len(x_pos) - win_width // 2):
            xs = x_pos[checkpoint - int(win_width / 2): checkpoint + int(win_width / 2)]
            ys = y_pos[checkpoint - int(win_width / 2): checkpoint + int(win_width / 2)]

            xs1 = xs[1: int(len(xs) / 2) + 1] - float(xs[1: int(len(xs) / 2) + 1][0])
            xs2 = xs[int(len(xs) / 2):] - float(xs[int(len(xs) / 2):][0])
            ys1 = ys[1: int(len(ys) / 2) + 1] - float(ys[1: int(len(ys) / 2) + 1][0])
            ys2 = ys[int(len(ys) / 2):] - float(ys[int(len(ys) / 2):][0])

            cum_xs1 = abs(np.cumsum(abs(xs1)))
            cum_xs2 = abs(np.cumsum(abs(xs2)))
            cum_ys1 = abs(np.cumsum(abs(ys1)))
            cum_ys2 = abs(np.cumsum(abs(ys2)))

            xs_max_val = max(np.max(abs(cum_xs1)), np.max(abs(cum_xs2)))
            cum_xs1 = cum_xs1 / xs_max_val
            cum_xs2 = cum_xs2 / xs_max_val

            ys_max_val = max(np.max(abs(cum_ys1)), np.max(abs(cum_ys2)))
            cum_ys1 = cum_ys1 / ys_max_val
            cum_ys2 = cum_ys2 / ys_max_val

            vals.append((abs(cum_xs1[-1] - cum_xs2[-1] + cum_ys1[-1] - cum_ys2[-1]))
                        + (max(np.std(xs1), np.std(xs2)) - min(np.std(xs1), np.std(xs2)))
                        + (max(np.std(ys1), np.std(ys2)) - min(np.std(ys1), np.std(ys2))))

        vals = np.concatenate((np.ones(int(win_width / 2)) * 0, vals))
        vals = np.concatenate((vals, np.ones(int(win_width / 2)) * 0))
        vals = np.array(vals)
        all_vals.append(vals)

    all_vals = np.array(all_vals) + 1e-5
    return all_vals


def slice_data(signal_seq, jump_d, ext_width, shift_width):
    slice_d = []
    indice = []
    for i in range(ext_width, signal_seq.shape[1] - ext_width, jump_d):
        crop = signal_seq[:, i - shift_width//2: i + shift_width//2]
        slice_d.append(crop)
        indice.append(i)
    return np.array(slice_d), np.array(indice) - ext_width


def local_maximum(signal, cp, seuil=5):
    while True:
        vals = [signal[x] if 0 <= x < signal.shape[0] else -1 for x in range(cp-seuil, cp+1+seuil)]
        if len(vals) == 0:
            return -1
        new_cp = cp + np.argmax(vals) - seuil
        if new_cp == cp:
            return new_cp
        else:
            cp = new_cp


def position_extension(x, y, ext_width):
    datas = []
    for data in [x, y]:
        delta_prev_data = -subtraction(data[:min(data.shape[0], ext_width)])[1:]
        delta_prev_data[0] += float(data[0])
        prev_data = np.cumsum(delta_prev_data)[::-1]

        delta_next_data = -subtraction(data[data.shape[0] - min(data.shape[0], ext_width):][::-1])[1:]
        delta_next_data[0] += float(data[-1])
        next_data = np.cumsum(delta_next_data)

        ext_data = np.concatenate((prev_data, data))
        ext_data = np.concatenate((ext_data, next_data))
        datas.append(ext_data)
    return np.array(datas), delta_prev_data.shape[0], delta_next_data.shape[0]


def s_shape(x, beta=3):
    x = np.minimum(np.ones_like(x)*0.999, x)
    x = np.maximum(np.ones_like(x)*0.001, x)
    return 1 / (1 + (x / (1-x))**(-beta))


def density_estimation(x, y, max_nb):
    densities = []
    dist_amp = 2.0
    local_mean_window_size = 5

    for i in range(x.shape[0]):
        density1 = 0
        density2 = 0

        slice_x = x[max(0, i - max_nb // 2):i].copy()
        slice_y = y[max(0, i - max_nb // 2):i].copy()

        if len(slice_x) > 0:
            mean_dist = np.sqrt(subtraction(slice_x) ** 2 + subtraction(slice_y) ** 2).mean()
            mean_dist *= dist_amp

            slice_x -= slice_x[len(slice_x) // 2]
            slice_y -= slice_y[len(slice_y) // 2]
            for s_x, s_y in zip(slice_x, slice_y):
                if np.sqrt(s_x ** 2 + s_y ** 2) < mean_dist:
                    density1 += 1

        slice_x = x[i:min(x.shape[0], i + max_nb // 2)].copy()
        slice_y = y[i:min(x.shape[0], i + max_nb // 2)].copy()

        if len(slice_x) > 0:
            mean_dist = np.sqrt(subtraction(slice_x) ** 2 + subtraction(slice_y) ** 2).mean()
            mean_dist *= dist_amp

            slice_x -= slice_x[len(slice_x) // 2]
            slice_y -= slice_y[len(slice_y) // 2]
            for s_x, s_y in zip(slice_x, slice_y):
                if np.sqrt(s_x ** 2 + s_y ** 2) < mean_dist:
                    density2 += 1
        densities.append(max(density1, density2))

    # local_mean
    new_densities = []
    for i in range(len(densities)):
        new_densities.append(np.mean(densities[max(0, i - local_mean_window_size // 2):
                                               min(len(densities), i + local_mean_window_size // 2 + 1)]))
    densities = new_densities
    return np.array(densities)


def signal_from_extended_data(x, y, win_widths, ext_width, jump_d, shift_width):
    assert ext_width > shift_width
    datas, shape_ext1, shape_ext2 = position_extension(x, y, ext_width)
    signal = make_signal(datas[0], datas[1], win_widths)

    density = density_estimation(datas[0], datas[1],
                                 max_nb=MAX_DENSITY_NB * 2)

    denoised_den = denoise_tv_chambolle(density, weight=3, eps=0.0002, max_num_iter=100, channel_axis=None)
    denoised_den = s_shape(denoised_den / MAX_DENSITY_NB)

    signal = signal[:, ] * denoised_den
    sliced_signals, slice_indice = slice_data(signal, jump_d, shape_ext1, shift_width)
    return sliced_signals


def local_roughness(signal, window_size):
    uc_signal = subtraction(signal)
    uc_signal /= abs(uc_signal)
    counts = []
    for i in range(window_size//2, len(uc_signal) - window_size//2):
        count = 0
        cur_state = 1
        for j in range(i-window_size//2, i+window_size//2):
            new_state = uc_signal[j]
            if new_state != cur_state:
                count += 1
            cur_state = new_state
        counts.append(count)
    counts = np.concatenate(([counts[0]] * (window_size//2), counts))
    counts = np.concatenate((counts, [counts[-1]] * (window_size//2)))
    return counts


def slice_normalize(slices):
    val = np.mean(np.sum(slices, axis=(2)).T, axis=0)
    val = val - np.min(val)
    val = val / np.max(val)
    return val


def partition_trajectory(x, y, cps):
    if len(cps) == 0:
        return [x], [y]
    new_x = []
    new_y = []
    for i in range(1, len(cps)):
        new_x.append(x[cps[i-1]:cps[i]])
        new_y.append(y[cps[i-1]:cps[i]])
    return new_x, new_y


def sort_by_signal(signal, cps):
    sort_indice = np.argsort(signal[cps])
    indice_tuple = [(i, i+1) for i in sort_indice]
    return indice_tuple, sort_indice


def predict_alphas(x, y, reg_model):
    pred_alpha = reg_model.alpha_predict(np.array([x, y]))
    return pred_alpha


def exhaustive_cps_search(x, y, win_widths, shift_width, ext_width, search_seuil=0.20,
                          cluster=None, cluster_states=None, reg_model=None):
    if len(x) < np.min(REG_MODEL_NUMS):
        if cluster is None:
            return np.array([0, len(x)]), np.array([1.0]), reg_model.k_predict([[x, y]]), np.array([2]), np.array([len(x)])
        else:
            """
            k_preds, alpha_preds, states = post_processing(np.array([0, len(x)]), [np.mean(cluster.means_[:, 1])], [np.mean(cluster.means_[:, 0])], np.array([2]),
                                                           cluster, cluster_states,
                                                           ad_length=ADJUST_MIN_SEG_LENGTH, force_imm=True)
            return np.array([0, len(x)]), alpha_preds, k_preds, states, np.array([len(x)])
            """
            return np.array([0, len(x)]), np.array([np.nan]), reg_model.k_predict([[x, y]]), np.array([np.nan]), np.array([len(x)])

    if cluster is not None and len(cluster.means_) == 1:
        start_cps = []
        slice_norm_signal = np.zeros_like(x.shape)
    else:
        if len(x) + 2 * (len(x) - 1) >= win_widths[0]:
            sliced_signals = signal_from_extended_data(x, y, win_widths, ext_width, 1, shift_width)
            slice_norm_signal = slice_normalize(sliced_signals)

            det_cps = []
            for det_cp in np.where(slice_norm_signal > search_seuil)[0]:
                det_cps.append(local_maximum(slice_norm_signal, det_cp, seuil=3))
            det_cps = np.unique(det_cps)
            start_cps = list(det_cps.copy())
        else:
            start_cps = []
            slice_norm_signal = np.zeros_like(x.shape)

    start_cps.append(0)
    start_cps.append(len(x))
    start_cps = list(np.sort(start_cps))
    while True:
        short_segment_flag = 0
        sorted_indice_tuple, sorted_indice = sort_by_signal(slice_norm_signal, start_cps[1:-1])
        for i in sorted_indice:
            i += 1
            if (start_cps[i] - start_cps[i - 1] < np.min(REG_MODEL_NUMS) or
                    start_cps[i + 1] - start_cps[i] < np.min(REG_MODEL_NUMS)):
                start_cps.remove(start_cps[i])
                short_segment_flag = 1
                break
        if short_segment_flag == 0:
            break

    while True:
        filtered_cps = []
        alpha_preds = []
        k_inputs = []

        part_xs, part_ys = partition_trajectory(x, y, start_cps)
        for p_x, p_y in zip(part_xs, part_ys):
            alpha_pred = predict_alphas(p_x, p_y, reg_model)
            alpha_preds.append(alpha_pred)
            k_inputs.append([p_x, p_y])
        k_preds = reg_model.k_predict(k_inputs)

        delete_cps = -1
        if cluster is not None:
            sorted_indice_tuple, sorted_indice = sort_by_signal(slice_norm_signal, start_cps[1:-1])
            for (l, r), i in zip(sorted_indice_tuple, sorted_indice):
                i += 1

                cluster_pred_label = cluster.predict([[alpha_preds[l], k_preds[l]], [alpha_preds[r], k_preds[r]]])
                cluster_pred_proba = cluster.predict_proba([[alpha_preds[l], k_preds[l]], [alpha_preds[r], k_preds[r]]])

                prev_cluster_pred_label = cluster_pred_label[0]
                after_cluster_pred_label = cluster_pred_label[1]
                prev_cluster_pred_probas = cluster_pred_proba[0]
                after_cluster_pred_probas = cluster_pred_proba[1]

                left_length = start_cps[i] - start_cps[i - 1]
                right_length = start_cps[i + 1] - start_cps[i]

                del_conditions = DEL_CONDITIONS
                len_conds = [-1, -1]

                for cond_k, length in enumerate([left_length, right_length]):
                    if length < 16:
                        len_conds[cond_k] = 0
                    elif length < 32:
                        len_conds[cond_k] = 1
                    elif length < 64:
                        len_conds[cond_k] = 2
                    else:
                        len_conds[cond_k] = 3

                if after_cluster_pred_probas[prev_cluster_pred_label] > del_conditions[len_conds[1]] and \
                        prev_cluster_pred_probas[after_cluster_pred_label] > del_conditions[len_conds[0]]:
                    delete_cps = start_cps[i]

                if delete_cps != -1:
                    break

        if delete_cps == -1:
            filtered_cps = start_cps
            break
        else:
            start_cps.remove(delete_cps)

    seg_lengths = subtraction(np.array(filtered_cps))[1:]
    alpha_preds = np.array(alpha_preds)
    filtered_cps = np.array(filtered_cps)
    k_preds = np.array(k_preds)
    states = []

    if cluster is not None:
        pred_set = np.array([alpha_preds, k_preds]).T
        for alpha, label in zip(alpha_preds, cluster.predict(pred_set)):
            states.append(label)
        """
        k_preds, alpha_preds, states = post_processing(filtered_cps, k_preds, alpha_preds, states,
                                                       cluster, cluster_states,
                                                       ad_length=ADJUST_MIN_SEG_LENGTH, force_imm=True)
        """
    alpha_preds = np.minimum(np.ones_like(alpha_preds) * 1.999, np.maximum(np.ones_like(alpha_preds) * 0.001,
                                                                           np.array(alpha_preds)))
    return filtered_cps, alpha_preds, k_preds, states, seg_lengths


def post_processing(cps, ks, alphas, states, cluster, cluster_states, ad_length, force_imm=False):
    adjusted_alphas = []
    adjusted_ks = []
    adjusted_states = []

    pred_set = np.array([alphas, ks]).T
    for i, (probas, label) in enumerate(zip(cluster.predict_proba(pred_set), cluster.predict(pred_set))):
        if cps[i + 1] - cps[i] < ad_length:
            adjusted_alphas.append(cluster.means_[label][0])
            adjusted_ks.append(cluster.means_[label][1])
            adjusted_states.append(cluster_states[label])
        else:
            adjusted_alphas.append(alphas[i])
            adjusted_ks.append(ks[i])
            adjusted_states.append(states[i])

    if force_imm:
        for i in range(len(adjusted_states)):
            if adjusted_states[i] == 0:
                adjusted_alphas[i] = 1e-3

    return adjusted_ks, adjusted_alphas, adjusted_states


def density_measurement(center, samples, rad=0.15):
    nb_dense = 0
    for samp in samples:
        if np.sqrt((center[0] - samp[0]) ** 2 + (center[1] - samp[1]) ** 2) < rad:
            nb_dense += 1
    return nb_dense


def cluster_define(samples, state_number):
    param_grid = {
        "n_components": CLUSTER_RANGE,
    }
    grid_search = GridSearchCV(
        GaussianMixture(max_iter=2000, n_init=20, covariance_type='tied'), param_grid=param_grid,
        scoring=gmm_bic_score
    )
    grid_search.fit(samples)
    cluster_df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "mean_test_score"]
    ]
    cluster_df["mean_test_score"] = -cluster_df["mean_test_score"]
    cluster_df = cluster_df.rename(
        columns={
            "param_n_components": "Number of components",
            "mean_test_score": "BIC score",
        }
    )
    opt_nb_component = np.argmin(cluster_df["BIC score"]) + param_grid['n_components'][0]
    selected_nb = state_number
    diag_ = False
    print(f'Estimated nb of states: {opt_nb_component}')
    print(f'Fixed nb of states: {selected_nb}')

    if selected_nb == -1:
        cluster = BayesianGaussianMixture(n_components=opt_nb_component, max_iter=2000, n_init=20,
                                          weight_concentration_prior=1e7, mean_precision_prior=1e-7,
                                          covariance_type='tied').fit(samples)
    elif opt_nb_component < selected_nb or selected_nb == 1:
        cluster = BayesianGaussianMixture(n_components=selected_nb, max_iter=2000, n_init=20,
                                          weight_concentration_prior=1e7, mean_precision_prior=1e-7,
                                          covariance_type='tied').fit(samples)
    else:
        cluster = BayesianGaussianMixture(n_components=opt_nb_component, max_iter=4000, n_init=20,
                                          mean_precision_prior=1e-7,
                                          covariance_type='diag').fit(samples)
        spec_cluster = BayesianGaussianMixture(n_components=selected_nb, max_iter=2000, n_init=30,
                                               weight_concentration_prior=1e7, mean_precision_prior=1e-7,
                                               covariance_type='tied').fit(samples)
        label_set = set()
        turn_on_indices = []
        meas_den = []
        for mean_ in cluster.means_:
            meas_den.append(density_measurement(mean_, samples, rad=0.15))

        order = np.argsort(meas_den)[::-1]
        for cov_, mean_, index in zip(cluster.covariances_[order], cluster.means_[order], np.arange(opt_nb_component)[order]):
            lb = spec_cluster.predict(mean_.reshape(-1, 2))[0]
            if lb not in label_set:
                turn_on_indices.append(index)
                label_set.add(lb)

        cov_weights = cluster.weights_
        for i in range(len(cov_weights)):
            if i not in turn_on_indices:
                cov_weights[i] = 0.
            else:
                cov_weights[i] = 1.

        for i, c_mean in enumerate(cluster.means_):
            if c_mean[0] < 0.3 or c_mean[0] > 1.75 and cov_weights[i] >= 0.99:
                diag_ = True

        if diag_:
            updated_weights = cov_weights / np.sum(cov_weights)
            cluster.weights_ = updated_weights
            resampled = np.vstack((cluster.sample(20000)[0], samples))
            cluster = BayesianGaussianMixture(n_components=selected_nb, max_iter=3000, n_init=20,
                                              covariance_type='diag',
                                              mean_precision_prior=1e-7).fit(resampled)
        else:
            cluster = spec_cluster

    if diag_:
        cluster_states = []
        for i in range(len(cluster.means_)):
            if cluster.means_[i][1] < -1.5 and cluster.covariances_[i][1] < 0.05 and cluster.means_[i][0] < 0.45:
                cluster_states.append(0)
            elif cluster.means_[i][0] < 0.5 and cluster.covariances_[i][1] >= 0.05:
                cluster_states.append(1)
            elif cluster.means_[i][0] > 1.75:
                cluster_states.append(3)
            else:
                cluster_states.append(2)
    else:
        cluster_states = [2 for _ in range(len(cluster.means_))]

    print('Cluster centers: ', cluster.means_)
    print('Cluster weights: ', cluster.weights_)
    print('Cluster covs: ', cluster.covariances_)
    return cluster, cluster_states


def main(public_data_path, path_results, image_path, make_image):
    METADATA_ID = np.random.randint(0, 1024 * 1024)
    WIN_WIDTHS = np.arange(20, 40, 2)
    SHIFT_WIDTH = 5

    states_nb = load_priors.load_state_prior(path_results)
    all_alphas, all_ks, all_seg_lengths = load_priors.load_priors(path_results)
    all_alphas = all_alphas[np.argwhere(all_seg_lengths > CLUSTER_SEG_LENGTH_SEUIL).flatten()]
    all_ks = all_ks[np.argwhere(all_seg_lengths > CLUSTER_SEG_LENGTH_SEUIL).flatten()]
    all_seg_lengths = all_seg_lengths[np.argwhere(all_seg_lengths > CLUSTER_SEG_LENGTH_SEUIL).flatten()]

    if np.sum(all_alphas > 1.8) > 10:
        mean_k_ = np.mean(all_ks[all_alphas > 1.8])
        std_k_ = np.std(all_ks[all_alphas > 1.8])
        all_ks[all_alphas > 1.8] = np.random.normal(loc=mean_k_, scale=std_k_ / 1.5,
                                                    size=len(all_ks[all_alphas > 1.8]))

    stack_samples = np.array([all_alphas, all_ks]).T
    print('Nb of samples: ', len(stack_samples))

    cluster, cluster_states = cluster_define(stack_samples, states_nb)
    if make_image:
        sample_distribution(image_path, stack_samples, all_seg_lengths, cluster, cluster_states)
        cluster_boundary(image_path, all_alphas, all_ks, cluster)
        print(f'Cluster images are generated...')

    dfs, file_names = load_datas(public_data_path)
    PBAR = tqdm(total=len(dfs), desc="BI-ADD", unit=f"file", ncols=120)
    reg_model = RegModel(REG_MODEL_NUMS)
    for df, f_name in zip(dfs, file_names):
        andi_outputs = ''
        result_file = path_results + f'{f_name}_biadd.h5'
        andi_submission_file = path_results + f'{f_name}.txt'
        
        if not os.path.exists(result_file):
            traj_idx = np.sort(df.traj_idx.unique())
            df_xs = []
            df_ys = []
            df_zs = []
            df_index = []
            df_frames = []
            df_ks = []
            df_alphas = []
            df_states = []
            for idx in traj_idx:
                frames = np.array(df[df.traj_idx == idx])[:, 1]
                x = np.array(df[df.traj_idx == idx])[:, 2]
                y = np.array(df[df.traj_idx == idx])[:, 3]
                if 'z' in df:
                    z = np.array(df[df.traj_idx == idx])[:, 4]
                else:
                    z = np.zeros_like(x)

                cps, alphas, ks, states, seg_lengths = exhaustive_cps_search(x, y, WIN_WIDTHS, SHIFT_WIDTH,
                                                                            EXT_WIDTH,
                                                                            search_seuil=SEARCH_SEUIL,
                                                                            cluster=cluster,
                                                                            cluster_states=cluster_states,
                                                                            reg_model=reg_model)

                prediction_traj = [idx.astype(int)]
                for k, alpha, state, cp in zip(ks, np.round(alphas, 5), states, cps[1:]):
                    prediction_traj.append(np.round(10 ** k, 8))
                    prediction_traj.append(alpha)
                    prediction_traj.append(state)
                    prediction_traj.append(cp)

                #if idx == 4336:
                #    cps_visualization(image_path, x, y, cps, alpha=0.8, ext=50)

                andi_outputs += ','.join(map(str, prediction_traj))
                andi_outputs += '\n'

                prev_cp = 0
                for k, alpha, state, cp in np.array(prediction_traj[1:]).reshape(-1, 4):
                    df_ks.extend([round(k, 5)] * int(cp - prev_cp))
                    df_alphas.extend([round(alpha, 5)] * int(cp - prev_cp))
                    df_states.extend([state] * int(cp - prev_cp))
                    prev_cp = cp

                df_xs.extend(list(x))
                df_ys.extend(list(y))
                df_zs.extend(list(z))
                df_frames.extend(list(frames))
                df_index.extend([idx] * len(x))

            with open(andi_submission_file, 'w') as f:
                f.write(andi_outputs)
                f.close()

            result_df = pd.DataFrame({'traj_idx':df_index,
                                      'frame':df_frames,
                                      'x':df_xs,
                                      'y':df_ys,
                                      'z':df_zs,
                                      'state':df_states,
                                      'K':df_ks,
                                      'alpha':df_alphas})
            result_df.attrs.update({
                'sample_id':METADATA_ID,
                'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            with pd.HDFStore(result_file, 'w') as hdf_store:
                hdf_store.put('data', result_df, format='table')
                hdf_store.get_storer('data').attrs.metadata = result_df.attrs

        else:   
            print(f'Result already exists: data:{f_name}')

        PBAR.update(1)
    PBAR.close()

if __name__ == "__main__":
    # data_path, result_path, image_path, True
    main(sys.argv[1], sys.argv[2], sys.argv[3], bool(sys.argv[4]))
