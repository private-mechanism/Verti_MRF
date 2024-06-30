import os
import time
import argparse



thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from exp.evaluate import run_experiment, split
from components.utils.preprocess import preprocess

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--paradigm', type=str, default = 'ver_PrivMRF')
parser.add_argument('--private_method', type=str, default = 'fmsketch') # random_response/fmsketch
parser.add_argument('--epsilon', type=float, default=0.8)
parser.add_argument('--task', type=str, default='tvd') #tvd/svm
parser.add_argument('--dataset', type=str, default='nltcs')

if __name__ == '__main__':
    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)
    
    args = parser.parse_args()

    preprocess('nltcs')

    data = args.dataset
    method = args.paradigm
    exp = args.task
    epsilon = args.epsilon
    private_method= args.private_method


    repeat = 5
    exp_name = private_method

    if exp == 'svm':
        run_experiment(data, method, exp_name, task='SVM',private_method=private_method, epsilon=epsilon, repeat=repeat, classifier_num=5, generate=True)
        run_experiment(data, method, exp_name, task='SVM', private_method=private_method,epsilon=epsilon, repeat=repeat, classifier_num=1, generate=False)
    else:
        run_experiment(data, method, exp_name, task='TVD', private_method=private_method, epsilon=epsilon, repeat=repeat, classifier_num=25)


