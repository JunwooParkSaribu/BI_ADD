import os
import sys
import subprocess
import sklearn
import skimage
import numpy
import pandas
from modules import load_priors


PYTHON_VERSION = 'python3'
PUBLIC_DATA_PATH = f'inputs/'
RESULT_DIR_PATH = f'outputs/'


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def main(data_path, result_path):
    data_list = []
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
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            data_list.append(file)

    load_priors.build_state_priror(result_path, data_list)
    print('State numbers are loaded')

    print(f'Prior building on the data...')
    proc = run_command([PYTHON_VERSION, f'./modules/build_priors.py',
                        f'{data_path}', f'{result_path}', f'{image_path}',
                        f'{False}'])
    proc.wait()
    if proc.poll() == 0:
        print(f'-> successfully finished')
    else:
        print(f'-> failed with status:{proc.poll()}')


    print(f'Prediction on the data...')
    proc = run_command([PYTHON_VERSION, f'ana_prediction.py',
                        f'{data_path}', f'{result_path}', f'{image_path}',
                        f'{False}'])
    proc.wait()
    if proc.poll() == 0:
        print(f'-> successfully finished')
    else:
        print(f'-> failed with status:{proc.poll()}')


if __name__ == "__main__":
    main(PUBLIC_DATA_PATH, RESULT_DIR_PATH)
