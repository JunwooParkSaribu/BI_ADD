import os
import sys
import subprocess
import sklearn
import skimage
import numpy
import pandas
from modules import load_priors


N_EXPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
TRACKS = [1, 2]
submit_number = 0
PYTHON_VERSION = 'python3'
PUBLIC_DATA_PATH = f'public_data_challenge_v0/'
RESULT_DIR_PATH = f'result_final_{submit_number}/'


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def main(data_path, result_path):
    try:
        if not os.path.exists(f'./models'):
            raise Exception('Models are not found, please download pretrained models')
        for model_num in [5, 8, 12, 16, 32, 64, 128]:
            if not os.path.exists(f'./models/reg_model_{model_num}.keras'):
                raise Exception(f'reg_model_{model_num}.keras doesn\'t exist')
        if not os.path.exists(f'./models/reg_k_model.keras'):
            raise Exception(f'reg_k_model.keras doesn\'t exist')

        run_command([PYTHON_VERSION, f'1 + 1'])
        print(f'numpy version: {numpy.__version__}')
        print(f'pandas version: {pandas.__version__}')
        print(f'sklearn version: {sklearn.__version__}')
        print(f'skimage version: {skimage.__version__}\n')
    except Exception as e:
        print(f'*** Err msg: {e} ***')
        print('Following packages need to be installed')
        print('Python==3.10 or higher')
        print('Tensorflow==2.14.1')
        print('Latest version of sci-kit learn')
        print('Latest version of sci-kit image')
        print('Latest version of numpy')
        print('Latest version of pandas')
        print('---------------------------------------')
        sys.exit(1)

    image_path = result_path + f'images'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for track in TRACKS:
        path_track = result_path + f'track_{track}/'
        if not os.path.exists(path_track):
            os.makedirs(path_track)
        for exp in N_EXPS:
            path_exp = path_track + f'exp_{exp}/'
            if not os.path.exists(path_exp):
                os.makedirs(path_exp)

    load_priors.build_state_priror(result_path, TRACKS, N_EXPS)
    print('State numbers are loaded')

    for track in TRACKS:
        for exp in N_EXPS:
            print(f'Prior building on the track:{track} exp:{exp}')
            proc = run_command([PYTHON_VERSION, f'./modules/build_priors.py',
                                f'{data_path}', f'{result_path}', f'{image_path}',
                                f'{track}', f'{exp}', f'{False}'])

            proc.wait()
            if proc.poll() == 0:
                print(f'-> successfully finished')
            else:
                print(f'-> failed with status:{proc.poll()}')

    for track in TRACKS:
        for exp in N_EXPS:
            print(f'Prediction on the track:{track} exp:{exp}')
            proc = run_command([PYTHON_VERSION, f'ana_prediction.py',
                                f'{data_path}', f'{result_path}', f'{image_path}',
                                f'{track}', f'{exp}', f'{False}'])
            proc.wait()
            if proc.poll() == 0:
                print(f'-> successfully finished')
            else:
                print(f'-> failed with status:{proc.poll()}')


if __name__ == "__main__":
    print(f'Submit number: {submit_number}')
    main(PUBLIC_DATA_PATH, RESULT_DIR_PATH)
