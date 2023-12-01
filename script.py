import os
import time
import argparse



thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

from exp.evaluate import run_experiment, split
from PrivMRF.preprocess import preprocess

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--paradigm', type=str, default = 'ver_PrivMRF')
parser.add_argument('--private_method', type=str, default = 'fmsketch') # random_response/fmsketch/vertigan
parser.add_argument('--epsilon', type=float, default=0.8)
parser.add_argument('--task', type=str, default='tvd') #tvd/svm
parser.add_argument('--dataset', type=str, default='adult')
# python3 script.py adult 1

if __name__ == '__main__':
    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)
    
    args = parser.parse_args()

    preprocess('nltcs')
    preprocess('adult')
    preprocess('br2000')
    preprocess('fire')


    data_list = [args.dataset]
    method = args.paradigm
    exp = args.task
    epsilon = args.epsilon
    # method = 'ver_PrivMRF'
    # exp = 'tvd'
    cen_binning = False

    
    private_method_list= [args.private_method]

    theta_list = [0.4]

    epsilon_list = [epsilon]

    for theta in theta_list:
        for private_method in private_method_list:
            repeat = 2
            now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            if method== 'cen_PrivMRF':
                exp_name = 'cen_PrivMRF'
                if cen_binning == True:
                    exp_name = 'cen_PrivMRF'+'_binning'
            elif method == 'ver_PrivMRF':
                exp_name = private_method+'_1130_com_'
            else:
                exp_name = private_method+'_1128_'
            if exp == 'svm':
                run_experiment(data_list, method, exp_name, theta,task='SVM',private_method=private_method, epsilon_list=epsilon_list, repeat=repeat, classifier_num=5, generate=True)
                # exp_name = ''
                run_experiment(data_list, method, exp_name, theta,task='SVM', private_method=private_method,epsilon_list=epsilon_list, repeat=repeat, classifier_num=1, generate=False)
            else:
                run_experiment(data_list, method, exp_name, theta,task='TVD', private_method=private_method, epsilon_list=epsilon_list, repeat=repeat, classifier_num=25)


