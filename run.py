# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg
import os
import components.utils.tools as tools
from client import Client
from server import Server
import numpy as np
import sys
# import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# thread number for numpy (when it runs on CPU)
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

from components.utils.preprocess import read_preprocessed_data
from components.utils.attribute_hierarchy import get_one_level_hierarchy



def run(data, domain, attr_hierarchy, exp_name, private_method, epsilon, task='TVD', p_config=None):
    default_config = {

        'beta5':        0.00,   # construct inner Bayesian network
        'data':         'nltcs',
        'theta':        6,
        'print':        True,
        'score':        'pairwsie_TVD', 
        # pairwsie_TVD is emperically better
        # 'score':        'pairwsie_MI',
        # 'score':        'pairwise_entropy',

        'score_R':                      False,
        'init_measure':                 0, 
                                        # 0 inner Bayesian Network 
                                        # 1 all n way measure
                                        # 2 clique measure
                                        # 3 empty measure

        'supplement_2way':              False,
        'attr_measure':                 False,
        'enable_attribute_hierarchy':   False,


        'last_estimation':              False,
        'init_model':                   True,
        'max_level_gap':                1,
        

        'estimation_iter_num':          3000,
        'print_interval':               500,


        'max_clique_size':              2e6,
        'global_clique_size':           4e6,
        'max_parameter_size':           4e6,
        'size_penalty':                 1e-8,


        'estimation_method':            'mirror_descent',
        'max_measure_attr_num':          20,
        'max_measure_attr_num_privBayes':5,
        'convergence_ratio':             1.3,
        'final_convergence_ratio':       0.7,
        'ed_step_num':                   8,


        'use_exp_mech':                 -1,      # do not use exponential mechanism to select measures    # 'use_exp_mech': 0.05,
        'structure_entropy':            False,   # marginal_noise will be set 0 to calculate the entropy of structures
        'noise_type':                   'normal',


        # only support normal
        'query_eps':                    0.1,
        'uniform_corre':                False,

        # the above settings are copied from the code of PrivMRF, the following are specific for vertiMRF

        'm':                            2000,   
        'gamma':                        1,
        'multithreads':                 40,
        'data_num_theta':               0.1,     # proportion of privacy budget for sanitizing the data number

        'attribute_binning':            True,   # not applicable to nltcs dataset since each attribute of nltcs dataset is binary
        'binning_num':                  4,
        'binning_method':              'dist',   # choice:dist (equal-width), freq (equal-frequency)
        'binning_theta':                0.2,     # proportion of privacy budget for binning

        'uniform_sampling':             False,   # 'uniform_sampling' or 'Hisrec' technique

        'local_MRF':                    True,  
        'combine_MRF':                  True,
        'local_MRF_theta':              0.4,  # proportion of privacy budget for generating the local mrfs
        'consistency':                  False, # whether to use the consistency enforcement technique
    }

    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)

    if default_config['use_exp_mech'] > 0:
        default_config['beta1'] = 0.12 # dependency graph, Markov network
        default_config['beta2'] = 0.55 # marginal distributions of initial marginals
        default_config['beta4'] = 0.33 # marginal distributions of newly selected marginals
    else:
        if default_config['init_measure'] == 3:
            default_config['beta1'] = 0.10 
            default_config['beta2'] = 0.0
            default_config['beta3'] = 0.10
            default_config['beta4'] = 0.80
        else:
            default_config['beta1'] = 0.10 # dependency graph, Markov network
            default_config['beta2'] = 0.50 # marginal distributions of initial marginals
            default_config['beta3'] = 0.10 # query L_1 norms
            default_config['beta4'] = 0.30 # marginal distributions of newly selected marginals
            default_config['t'] = 0.8
            # beta2, beta4 is no longer uesful, we use t to allocate budget for marginal distribution
            # we ensure that beta2 + beta4 = 1 - (beta1 + beta3)

    config = default_config.copy()

    if p_config is not None:
        for item in p_config:
            config[item] = p_config[item]

    if not config['print']:
        temp_stream = sys.stdout
        sys.stdout = open('./temp/log.txt', 'w')
    config['private_method'] = private_method


    config['theta1'] = config['theta']
    config['theta2'] = config['theta']

    config['epsilon'] = epsilon
    config['exp_name'] = exp_name
    if attr_hierarchy is None:
        attr_hierarchy = get_one_level_hierarchy(domain)

    seeds = np.random.randint(0, high=100000, size=config['m'])

    client_list=[]

    # if config['data'] == 'nltcs':
    config['attribute_binning'] = False
    config['max_clique_size'] = 2e6
    config['global_clique_size'] = 4e6
    attr_alice = [i for i in range(8)]
    attr_bob = [i+8 for i in range(8)]
    tuple_alice = tuple(attr_alice)
    tuple_bob = tuple([i+8 for i in range(8)])

    #Instantiate local data parties and a central server
    client_list.append(Client('Alice', data[:,tuple_alice], domain, attr_alice, seeds, config, epsilon, attr_hierarchy,gpu=True))
    client_list.append(Client('Bob', data[:,tuple_bob], domain, attr_bob, seeds, config, epsilon, attr_hierarchy, gpu=True))
    server = Server(data, [i for i in range(16)], seeds, attr_hierarchy, domain, config, gpu=True)
    msg_list = []
    for client in client_list:
        msg_list.append(client.upload_msg())

    server.recieve_msg(msg_list)    

    if config['private_method'] == 'fmsketch':
        private_statistics = server.fm_generate_private_statistics()
    else:
        private_statistics = {}


    model = server.build_global_mrf(private_statistics)

    if config['last_estimation']:
        model.config['convergence_ratio'] = 1.0
        model.config['estimation_iter_num'] = 5000
        model.mirror_descent()
    if not config['print']:
        sys.stdout.close()
        sys.stdout = temp_stream
    os.chdir(cwd)
    return model




def run_syn_ver(data_name, exp_name, private_method, epsilon, task='TVD'):
    p_config = {}
    p_config['data'] = data_name
    data, domain, attr_list = read_preprocessed_data(data_name, task)
    nvalues=[]
    for attr in range(len(domain)):
        nvalues.append([i for i in range(domain.dict[attr]['domain'])])
    read_from_out = False
    # reading from the generated data file
    if read_from_out:
        data, headings = tools.read_csv('./out/' + exp_name + '.csv', print_info=False)
        data_list = np.array(data, dtype=int)
    # generating synthetic data with algorithms
    else:
        model = run(data, domain, attr_list, exp_name, private_method, epsilon, task, p_config)
        data_list = model.synthetic_data('./out/' + exp_name +'.csv')
    return list(data_list)

