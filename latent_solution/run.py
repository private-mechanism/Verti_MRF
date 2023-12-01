
# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg
import os
import PrivMRF
import PrivMRF.utils.tools as tools
from PrivMRF.domain import Domain
from client import Client
from server import Server
import numpy as np
from networkx.readwrite import json_graph
import json
import numpy as np
import pickle
import time
import sys
import logging
import networkx as nx
import matplotlib.pyplot as plt


# thread number for numpy (when it runs on CPU)
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

from PrivMRF.preprocess import read_preprocessed_data, postprocess
from PrivMRF.my_attribute_graph import AttributeGraph
from PrivMRF.attribute_hierarchy import get_one_level_hierarchy
from PrivMRF.my_markov_random_field import MarkovRandomField
from PrivMRF.preprocess import preprocess


def run(data, domain, attr_hierarchy, exp_name, epsilon, task='TVD', p_config=None):
    default_config = {

        'beta5':        0.00,   # construct inner Bayesian network
        'data':         'nltcs',
        'theta':        6,
        'print':        True,
        'score':        'pairwsie_TVD', # pairwsie_TVD is emperically better
        # 'score':        'pairwsie_MI',
        # 'score':        'pairwise_entropy',
        'score_R':                      False,
        'init_measure':                 0, # 0 inner Bayesian Network 
                                        # 1 all n way measure
                                        # 2 clique measure
                                        # 3 empty measure
        'supplement_2way':              False,
        'attr_measure':                 False,
        'enable_attribute_hierarchy':   False,
        # 'enable_attribute_hierarchy':   True,
        'last_estimation':              False,
        'init_model':                   True,
        'max_level_gap':                1,
        
        'estimation_iter_num':          3000,
        'print_interval':               500,

        'max_clique_size':              1e5,
        'global_clique_size':           1e5,
        'max_parameter_size':           1e5,
        'size_penalty':                 1e-8,

        'estimation_method':            'mirror_descent',

        'max_measure_attr_num':         20,
        'max_measure_attr_num_privBayes':5,

        'convergence_ratio':            1.3,
        'final_convergence_ratio':      0.7,

        'use_exp_mech':                 -1,      # do not use exponential mechanism to select measures
        # 'use_exp_mech':                 0.05,
        'structure_entropy':            False,   # marginal_noise will be set 0 to calculate the entropy of structures
        'noise_type':                   'normal',# only support normal
        'query_eps':                    0.1,
        'private_method':               'fmsketch', # choice: fmsketch, random_response
        'm':                            2000,
        'gamma':                        1,
        'multithreads':                 40,
        'attribute_binning':            True,
        'binning_num':                  2,
        'binning_method':               'dist', #choice:dist, freq
        'graph_est':                    'locally',
        'combine_method':               'consis_loss'
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

    # There might be no enough resource to run PrivMRF on GPU
    # acs should be runned on cpu, nltcs is too small and doesn't have to be runned on GPU
    gpu = False
    if config['data'] == 'adult' or config['data'] == 'br2000':
        gpu = True
    if config['data'] == 'acs' or config['data'] == 'nltcs':
        default_config['max_measure_attr_num'] = 10
        default_config['max_measure_attr_num_privBayes'] = 9
    if config['data'] == 'adult':
        default_config['enable_attribute_hierarchy'] = True

    config['theta1'] = config['theta']
    config['theta2'] = config['theta']
    config['epsilon'] = epsilon
    config['exp_name'] = 'PrivMRF_'+ config['data'] + '_' + exp_name +'_ver_epsilon' + str(epsilon)  +'_'
    if attr_hierarchy is None:
        attr_hierarchy = get_one_level_hierarchy(domain)
    print('PrivMRF')


    # res_list_list = []
    # for repeat in range(2):
    seeds = np.random.randint(0, high=100000, size=config['m'])

    client_list=[]
    # data1 = data[:,:7]
    # data2 = data[:,7:]
    if config['data'] == 'nltcs':
        attr_pairs = [[0,8],[0,9],[0,10],[0,11],[0,12],[0,13]]
        client_list.append(Client('Alice', data, domain, [0,1,2,3,4,5,6,7], seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data, domain, [8,9,10,11,12,13,14,15], seeds, config, epsilon,attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(16)], seeds, attr_hierarchy, domain, config, gpu=True)
    elif config['data'] == 'adult':
        # attr_pairs = [[2,8],[2,9],[2,10],[2,11],[2,12],[2,13]]
        client_list.append(Client('Alice', data, domain, [0,1,2,3,4,5,6,7], seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data, domain, [8,9,10,11,12,13,14], seeds, config, epsilon, attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(15)], seeds,  attr_hierarchy, domain, config, gpu=True)
    else:
        attr_pairs = [[0,8],[0,9],[0,10],[0,11],[0,12],[0,13],[0,14]]
        client_list.append(Client('Alice', data, domain, [0,1,2,3,4,5,6,7], seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data, domain, [8,9,10,11,12,13,14], seeds, config, epsilon,attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(16)], seeds, attr_hierarchy, domain, config, gpu=True)

    msg_list = []
    for client in client_list:
        # if config['attribute_binning']:
        #     client.binning()
        # if config['private_method'] == 'fmsketch':
        #     client.generate_multiple_sketches() 
        # else:
        #     client.generate_random_responses()
        msg_list.append(client.upload_msg())

    server.recieve_msg(msg_list)    
    # if config['private_method'] == 'random_response':
    #     server.rr_intersection_dic()

    # res_list = []
    # for attr_pair in attr_pairs:
    #     TVD = server.dp_TVD(attr_pair)
    #     res_list.append(TVD)
    # logging.info(f'the R-score of the specified attr pairs are {res_list}')

    if config['private_method'] == 'fmsketch':
        private_statistics = server.fm_generate_private_statistics()
    elif config['private_method'] == 'random_response':
        private_statistics = server.rr_generate_private_statistics()
    if 'graph_est' == 'globally':
        graph, measure_list, attr_hierarchy, attr_to_level, entropy = server.build_attribute_graph(private_statistics)
    # if 'graph_est' == 'locally':
    # graph, measure_list, attr_hierarchy, attr_to_level, entropy, adj = server.build_attribute_graph_local_graphs(private_statistics)

    # server.construct_mrf(graph, measure_list, attr_hierarchy, attr_to_level,private_statistics)
    server.combine_local_mrfs(private_statistics)
    # server.candidate_marginal_selection()
    # initialized_marginal_set = measure_list
    # logging.info(f'the initialized_marginal_set for epsilon {epsilon} is {initialized_marginal_set}')
    # finalized_marginal_set = server.refine_marginal_set(initialized_marginal_set)
    # logging.info(f'the finalized_marginal_set for epsilon {epsilon} is {finalized_marginal_set}')
    # exp_name = 'exp'
    # data_name = 'nltcs'
    # data_list = server.generate_data('./out/' + 'PrivMRF_'+ data_name + '_' + exp_name + '.csv')
    model = server.mrf
    if config['last_estimation']:
        model.config['convergence_ratio'] = 1.0
        model.config['estimation_iter_num'] = 5000
        model.mirror_descent()
        # MarkovRandomField.save_model(model, './temp/' + config['data'] + '_le_model.mrf')
    # time_cost = time.time() - start_time
    # print('time cost: {:.4f}s'.format(time_cost))
    if not config['print']:
        sys.stdout.close()
        sys.stdout = temp_stream

    os.chdir(cwd)
    return model

# used for experiments of the paper Data synthesis via differentially private Markov random field
def run_syn_fm(data_name, exp_name, epsilon, task='TVD'):
    p_config = {}
    p_config['data'] = data_name
    data, domain, attr_hierarchy = read_preprocessed_data(data_name, task)
    model = run(data, domain, attr_hierarchy, exp_name, epsilon, task, p_config)
    data_list = model.synthetic_data('./out/' + 'PrivMRF_' + data_name + '_ver_epsilon' + str(epsilon)  +'_'+ exp_name + '.csv')
    # return data_list
    return data_list
# data_list
        
    # start_time = time.time()
    
    # print('theta:', config['theta'])
    # if config['init_model']:
    #     # init_model = MyAttributeGraph(data, domain, attr_hierarchy, config, config['data'])
    #     # graph, measure_list, attr_hierarchy, attr_to_level, entropy = init_model.construct_model()'
    #     init_model = MyAttributeGraph(self.domain, self.noisy_data_num, self.attr_list, self.config, self.config['data'],self.fmsketches, self.eps)
    #     graph, measure_list, attr_hierarchy, attr_to_level, entropy = init_model.construct_model()
    #     # AttributeGraph.save_model(init_model, './temp/' + config['data'] + '_model.mdl')

    # # return entropy

    # # init_model = AttributeGraph.load_model('./temp/' + config['data'] + '_model.mdl')
    # graph = init_model.graph
    # measure_list = init_model.measure_list
    # attr_hierarchy = init_model.attr_list
    # attr_to_level = init_model.attr_to_level
    # data_num = init_model.data_num

    # model = MarkovRandomField(data, domain, graph, measure_list, \
    #     attr_hierarchy, attr_to_level, data_num, config, gpu=gpu)
    # model.entropy_descent()
    # # MarkovRandomField.save_model(model, './temp/' + config['data'] + '_model.mrf')

    # # model = MarkovRandomField.load_model('./temp/' + config['data'] + '_model.mrf')
    # if config['last_estimation']:
    #     model.config['convergence_ratio'] = 1.0
    #     model.config['estimation_iter_num'] = 5000
    #     model.mirror_descent()
    #     # MarkovRandomField.save_model(model, './temp/' + config['data'] + '_le_model.mrf')

    # time_cost = time.time() - start_time
    # print('time cost: {:.4f}s'.format(time_cost))

    # if not config['print']:
    #     sys.stdout.close()
    #     sys.stdout = temp_stream

    # os.chdir(cwd)


if __name__ == '__main__':
    # should provide int data
    data, _ = tools.read_csv('./preprocess/nltcs.csv')
    data = np.array(data, dtype=int)

    # domain of each attribute should be [0, 1, ..., max_value-1]
    # attribute name should be 0, ..., column_num-1.
    json_domain = tools.read_json_domain('./preprocess/nltcs.json')
    domain = Domain(json_domain, list(range(data.shape[1])))

    # you may set hyperparameters or specify other settings here
    config = {
    }

    # train a PrivMRF, delta=1e-5
    # for other dp parameter delta, calculate the privacy budget 
    # with cal_privacy_budget() of ./PrivMRF/utils/tools.py 
    # and hard code the budget in privacy_budget() of ./PrivMRF/utils/tools.py 
    for epsilon in [0.1,0.2]:
        data_list = run(data, domain, attr_hierarchy=None, exp_name='exp', epsilon=1.920, task='TVD', p_config=None)

    # model = PrivMRF.run(data, domain, attr_hierarchy=None, \
    #     exp_name='exp', epsilon=0.8, p_config=config)

    # generate synthetic data
    # syn_data = model.synthetic_data('./out.csv')