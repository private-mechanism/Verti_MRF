
# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg
import os
import Utils.utils.tools as tools
from Utils.domain import Domain
from client import Client
from server import Server
import numpy as np
from networkx.readwrite import json_graph
import json
import numpy as np
import sys
import logging
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# import torch
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# thread number for numpy (when it runs on CPU)
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

from Utils.preprocess import read_preprocessed_data, postprocess
from Utils.GraphCom_my_attribute_graph import AttributeGraph
from Utils.attribute_hierarchy import get_one_level_hierarchy
from Utils.OptMRF_markov_random_field import MarkovRandomField
from Utils.preprocess import preprocess



def run(data, domain, attr_hierarchy, exp_name, theta, private_method, epsilon, task='TVD', p_config=None):
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
        # 'private_method':               'random_response', 
        # Choices: fmsketch, random_response, latent_tree, verti_gan, scalar_product`
        'uniform_corre':                False,

        'm':                            2000,    # 0.07: 6682378.196367457    0.4:1449718.506370042#
        'gamma':                        1,
        'multithreads':                 40,
        'same_seed':                    True,    # 1730857.066429753, 1896907.9569721755(different seed) / 748258.1636430555, 1145925.308329834(same seed)
        'data_num_theta':               0.1,

        'attribute_binning':            True,
        'binning_num':                  4,
        'binning_method':              'dist',   # choice:dist, freq
        'binning_theta':                0.2,
        'use_binning2consis':           True,


        'graph_est':                   'globally',
        'combine_method':              'consis_loss',
        'binary':                       False, 
        'uniform_sampling':             False,

        'group_size':                    2,

        'local_MRF':                    True,
        'local_MRF_theta':              0.8,
        'weight_consis':                False,
        'combine_MRF':                  True,
        'consistency':                  False,

        'party_num':                    2
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

    config['local_MRF_theta'] = theta

    # There might be no enough resource to run PrivMRF on GPU
    # acs should be runned on cpu, nltcs is too small and doesn't have to be runned on GPU
    gpu = False
    if config['data'] == 'adult' or config['data'] == 'br2000' or config['data'] == 'fire':
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

    seeds = np.random.randint(0, high=100000, size=config['m'])

    client_list=[]

    if config['private_method'] == 'verti_gan':
        from vertigan import Generator,Discriminator,Verti_GAN
        from datasets import CategoricalDataset
        latent_dim = 32
        total_attr_list=[]
        alice_attr_list=[]
        bob_attr_list=[]
        if config['data'] == 'nltcs':
            # 定义超参数
            total_attr_list = [i for i in range(16)]
            alice_attr_list = [i for i in range(8)]
            bob_attr_list = [i+8 for i in range(8)]
        elif config['data'] == 'fire':
            # 定义超参数
            total_attr_list = [i for i in range(15)]
            alice_attr_list = [i for i in range(8)]
            bob_attr_list = [i+8 for i in range(7)]
        elif config['data'] == 'adult':
            # 定义超参数
            total_attr_list = [i for i in range(15)]
            alice_attr_list = [i for i in range(8)]
            bob_attr_list = [i+8 for i in range(7)]
        elif config['data'] == 'br2000':
            # 定义超参数
            total_attr_list = [i for i in range(14)]
            alice_attr_list = [i for i in range(7)]
            bob_attr_list = [i+7 for i in range(7)]
        
        
        latent_dim = np.sum([domain.project((attr,)).size() for attr in total_attr_list])
        # domain.project(tuple(total_attr_list)).size()
        alice_dim = np.sum([domain.project((attr,)).size() for attr in alice_attr_list])
        bob_dim = np.sum([domain.project((attr,)).size() for attr in bob_attr_list])
        output_dims = [alice_dim, bob_dim]  # 每个分支的输出维度
        party2attr = [tuple([i for i in range(alice_dim)]),tuple([i+alice_dim for i in range(bob_dim)])] 
            
        cuda = True if torch.cuda.is_available() else False

        def noise_function(n):
            # return 0
            return torch.randn(n, latent_dim)
        
        # 创建生成器和判别器实例
        real_data = pd.DataFrame(data)
        dataset = CategoricalDataset(real_data)
        multi_cate_dimensions_1 = dataset.dimensions[0:len(alice_attr_list)]
        multi_cate_dimensions_2 = dataset.dimensions[len(alice_attr_list):len(total_attr_list)]
        data_tensor = dataset.to_onehot_flat()
        global_generator = Generator(latent_dim, output_dims,\
                                     [multi_cate_dimensions_1,multi_cate_dimensions_2])
        # generator2 = Generator(latent_dim, output_dims)
        Alice_discriminator = Discriminator(output_dims[0])
        Bob_discriminator = Discriminator(output_dims[1])

        if cuda:
            global_generator.cuda()
            Alice_discriminator.cuda()
            Bob_discriminator.cuda()

        # discriminator3 = Discriminator(output_dims[2])

        # 创建生成器和判别器列表
        generator = global_generator
        local_discriminators = [Alice_discriminator, Bob_discriminator]
        # real_data = torch.tensor(data)
        Verti_GAN_model = Verti_GAN(generator,local_discriminators,noise_function,party2attr)
        Verti_GAN_model.train(data_tensor,epsilon*4, global_epochs=100,n_critics=10, batch_size=100,
              learning_rate=1e-4, clipping_bound=1)
        noise = torch.normal(mean=0,std=1,size=(len(data), latent_dim))
        flat_synth_data = Verti_GAN_model.generate(noise) 
        # flat_synth_data = Verti_GAN_model.recover(flat_synth_data)
        synth_data = dataset.from_onehot_flat(flat_synth_data)
        syn_data = synth_data.values
        return syn_data

    if config['data'] == 'nltcs':
        config['attribute_binning'] = False
        config['max_clique_size'] = 2e6
        config['global_clique_size'] = 4e6
        # if config['party_num'] == 2:
        attr_alice = [i for i in range(8)]
        attr_bob = [i+8 for i in range(8)]
        tuple_alice = tuple(attr_alice)
        tuple_bob = tuple([i+8 for i in range(8)])
        client_list.append(Client('Alice', data[:,tuple_alice], domain, attr_alice, seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data[:,tuple_bob], domain, attr_bob, seeds, config, epsilon, attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(16)], seeds, attr_hierarchy, domain, config, gpu=True)
    elif config['data'] == 'adult':
        attr_alice = [i for i in range(8)]
        attr_bob = [i+8 for i in range(7)]
        tuple_alice = tuple(attr_alice)
        tuple_bob = tuple([i+8 for i in range(7)])
        client_list.append(Client('Alice', data[:,tuple_alice], domain,  attr_alice , seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data[:,tuple_bob], domain, attr_bob, seeds, config, epsilon, attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(15)], seeds,  attr_hierarchy, domain, config, gpu=True)
    elif config['data'] == 'fire':
        attr_alice = [i for i in range(8)]
        attr_bob = [i+8 for i in range(7)]
        tuple_alice = tuple([i for i in range(8)])
        tuple_bob = tuple([i+8 for i in range(7)])
        client_list.append(Client('Alice', data[:,tuple_alice], domain,  attr_alice , seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data[:,tuple_bob], domain, attr_bob, seeds, config, epsilon, attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(15)], seeds,  attr_hierarchy, domain, config, gpu=True)
    elif config['data'] == 'acs':
        config['attribute_binning'] = False
        attr_alice = [i for i in range(11)]
        attr_bob = [i+11 for i in range(12)]
        tuple_alice = tuple([i for i in range(11)])
        tuple_bob = tuple([i+11 for i in range(12)])
        client_list.append(Client('Alice', data[:,tuple_alice], domain,  attr_alice , seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data[:,tuple_bob], domain,  attr_bob, seeds, config, epsilon, attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(23)], seeds,  attr_hierarchy, domain, config, gpu=True)
    elif config['data'] == 'br2000':
        attr_alice = [i for i in range(7)]
        attr_bob = [i+7 for i in range(7)]
        tuple_alice = tuple([i for i in range(7)])
        tuple_bob = tuple([i+7 for i in range(7)])
        client_list.append(Client('Alice', data[:,tuple_alice], domain, attr_alice , seeds, config, epsilon, attr_hierarchy,gpu=True))
        client_list.append(Client('Bob', data[:,tuple_bob], domain, attr_bob, seeds, config, epsilon, attr_hierarchy, gpu=True))
        server = Server(data, [i for i in range(14)], seeds,  attr_hierarchy, domain, config, gpu=True)
    msg_list = []
    for client in client_list:
        msg_list.append(client.upload_msg())

    server.recieve_msg(msg_list)    

    if config['private_method'] == 'fmsketch':
        private_statistics = server.fm_generate_private_statistics()
        # private_statistics = {}
    elif config['private_method'] == 'random_response':
        private_statistics = {}
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

# used for experiments of the paper Data synthesis via differentially private Markov random field
def run_syn_ver(data_name, exp_name, theta, private_method, epsilon, task='TVD'):
    p_config = {}
    p_config['data'] = data_name

    logging.info(f'###################################################-start!')
    
    data, domain, attr_list = read_preprocessed_data(data_name, task)
    # private_method = 'latent_tree'
    if private_method != 'verti_gan':
        nvalues=[]
        for attr in range(len(domain)):
            nvalues.append([i for i in range(domain.dict[attr]['domain'])])
        read_from_out = False
        if read_from_out:
            data, headings = tools.read_csv('./out/' + 'PrivMRF_' + data_name + '_ver_epsilon' + str(epsilon)  +'_'+ exp_name + '.csv', print_info=False)
            data_list = np.array(data, dtype=int)
        else:
            model = run(data, domain, attr_list, exp_name, theta, private_method, epsilon, task, p_config)
            data_list = model.synthetic_data('./out/' + 'PrivMRF_' + data_name + '_ver_epsilon' + str(epsilon)  +'_'+ exp_name +'.csv')
    else:
        read_from_out = False
        if read_from_out:
            data, headings = tools.read_csv('./out/' + 'PrivMRF_' + data_name + '_ver_epsilon' + str(epsilon)  +'_'+ exp_name + '.csv', print_info=False)
            data_list = np.array(data, dtype=int)
        else:
            data_list = run(data, domain, attr_list, exp_name, theta, private_method, epsilon, task, p_config)
            path = './out/' + 'PrivMRF_' + data_name + '_ver_epsilon' + str(epsilon)  +'_'+ exp_name + '.csv'
            tools.write_csv(list(data_list), list(range(len(domain))), path)
    logging.info(f'###################################################-end!')
    return list(data_list)

