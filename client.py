import Utils
import Utils.utils.tools as tools
from Utils.domain import Domain
from Utils.fmsketch import complement_fm_sketch
from Utils.utils import corex
from Utils.utils.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from Utils.LocMRF_attribute_graph import AttributeGraph
from Utils.LocMRF_markov_random_field import MarkovRandomField

import numpy as np 
import pandas as pd
import concurrent.futures
import logging
import time
import itertools
import random
import code
from scipy import stats
import copy
import json
import pickle

class Client:
    def __init__(self, ID, data, domain, attr_list, seeds,\
     config, eps, attr_hierarchy, gpu=True):
        self.ID = ID
        self.data = data
        self.domain = domain
        self.attr_list = attr_list
        # self.attr_to_level = attr_to_level
        self.config = config
        self.data_num = len(self.data)
        if self.config['structure_entropy']:
            self.noisy_data_num = self.data_num
        # self.max_measure_attr_num = config['max_measure_attr_num']
        self.attr_num = len(self.attr_list)
        self.gpu = gpu
        self.private_statistics = {}
        self.bin_cut ={}
        for attr in self.attr_list:
            self.bin_cut[attr] = self.config['binning_num']
        self.eps = eps
        self.bin_num = self.config['binning_num']
        self.seeds = seeds
        self.binning_map = {}  # {attr:{domain:(21,2), cut:5},...}
        self.cut_map = {}
        self.attr_hierarchy = attr_hierarchy
        self.bin_2_original = {}
        # for index, attr in enumerate(self.attr_list):
        #     self.attr_hierarchy.append(attr_hierarchy[attr])


        #########################################allocate the privacy budget #######################################

        self.proportion_for_local_MRF = 0
        self.proportion_for_binning = 0
        self.proportion_for_data_num = 0
        if self.config['local_MRF']:
            data_parties = 2
            self.proportion_for_local_MRF = self.config['local_MRF_theta']
            self.budget_for_local_MRF = self.eps*self.proportion_for_local_MRF/data_parties
            logging.info(f'the privbacy budget for generate local MRF for {self.ID} is {self.budget_for_local_MRF}')


        self.budget_for_binning = 0
        if self.config['attribute_binning'] and not self.config['uniform_sampling']:
            self.proportion_for_binning = self.config['binning_theta']
            count = 0
            for attr in range(len(self.domain)):
                if self.domain.dict[attr]['domain'] > self.config['binning_num']:
                    count += 1
            if self.config['use_binning2consis']:
                self.budget_for_binning = self.eps*self.proportion_for_binning/len(domain)
            else:
                self.budget_for_binning = self.eps*self.proportion_for_binning/count
        
        if self.config['private_method'] == 'fmsketch':
            self.proportion_for_data_num = self.config['data_num_theta']
            self.budget_for_data_num = self.eps*self.proportion_for_data_num
            self.eps = self.eps*(1-self.proportion_for_local_MRF-self.proportion_for_binning-self.proportion_for_data_num)/np.sqrt(len(self.domain))
        else:
            self.eps = 2*self.eps*(1-self.proportion_for_local_MRF-self.proportion_for_binning)/len(self.domain)
    
        if self.config['private_method'] == 'fmsketch':
            logging.info(f'budget for binning is {self.budget_for_binning*len(domain)}, budget for local MRF is {self.budget_for_local_MRF*2}, \
                budget for noisy data num{self.budget_for_data_num},for generating sketches is {self.eps*np.sqrt(len(self.domain))}')
        else:
            logging.info(f'budget for binning is {self.budget_for_binning}, budget for local MRF is {self.budget_for_local_MRF*np.sqrt(2)}, \
                budget for LDP is {self.eps*len(self.domain)}')
            

        attr_list = [i for i in range(len(self.attr_list))]
        domain_temp = copy.copy(self.domain)
        new_dict = {key-self.attr_list[0]: domain_temp.dict[key] for key in self.attr_list}
        new_attr_list = [attr for attr in attr_list]
        self.domain_client = Domain(new_dict,new_attr_list)
        shape = [self.domain_client.dict[attr]['domain'] for attr in attr_list]
        logging.info(f'the shape of the attr list is {shape}')

    ############################################# intersection_clean ###############################################

    def clean_membership(self, attr):
        s = self.domain.dict[attr]['domain']
        clean_membership = []
        for i in range(s):
            clean_membership.append(set(np.where(self.data[:,attr-self.attr_list[0]] == i)[0]))
        return clean_membership
    
    def intersection_clean(self,attr_pair):
        splits=[]
        splits.append(self.clean_membership(attr_pair[0]))
        splits.append(self.clean_membership(attr_pair[1]))
        cartesian = list(itertools.product(*splits))
        clean_intersections = []
        for combine in cartesian:
            combine = [set(c) for c in combine]
            intersect = combine[0].intersection(*combine[1:])
            clean_intersections.append(intersect)
        clean_ca = [len(s) for s in clean_intersections]
        print("clean intersection size", clean_ca)
        print("sum:", np.sum(clean_ca))
        return clean_ca
    
    ################################################Sketch-based LocEnc

    def priv_gen_sketch(self, seed):
        '''
        Generate DP FM sketch for each attribute in the local set
        '''
        n = self.data_num
        gamma = self.config['gamma']
        temp_fmsketch={}
        priv_config = {'eps': self.eps, 'delta': 1 / n, 'm': self.config['m']}
        for attr in self.attr_list:
            memberships = self.clean_membership(attr)
            fm_sketch_for_attr, com_fm_sketch_for_attr = complement_fm_sketch(memberships, seed, gamma, priv_config)
            temp_fmsketch[attr] = {'private_statistics':com_fm_sketch_for_attr}
            if self.config['same_seed'] == False:
                # seed += int(attr)
                seed = np.random.randint(0, high=100000)
        return temp_fmsketch


    def generate_multiple_sketches(self):
        start_time = time.time()
        multithreads = self.config['multithreads']
        logging.info(f"multithreading, # of threads {multithreads}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=multithreads) as executor:
            future_to_seed_idx = {executor.submit(self.priv_gen_sketch,seed): idx for idx, seed in enumerate(self.seeds)}
            for future in concurrent.futures.as_completed(future_to_seed_idx):
                idx = future_to_seed_idx[future]
                try:
                    self.private_statistics[idx] = future.result()
                except Exception as exc:
                    print(f"Execption when running {idx}-th seed {self.seeds[idx]}: {exc}")
        seconds = time.time() - start_time
        print("multithreading, run time", seconds)
        

    #####################################################FO-based LocEnc

    def ldp_generate_random_responses(self):
        for attr in self.attr_list:
            perturbed_value = self.ldp_perturb_attr(attr)
            self.private_statistics[attr] = perturbed_value


    # domain_size = self.domain.dict[attr]['domain']
    def ldp_perturb_attr(self, attr):
        domain_size = self.domain.dict[attr]['domain']
        if self.ID == 'Bob':
            attr = attr - self.attr_list[0]
        real_values = self.data[:,attr]
        logging.info("===> using GRR for ldp")
        perturbed = rr_perturb(real_values, self.eps, domain_size)
        return perturbed
    
    
    def sanitize_histogram(self,histogram,epsilon,sensitivity):
        dim = histogram.ndim
        noise = np.random.laplace(0, sensitivity/ epsilon,dim)
        sanitized_histogram = histogram + noise
        sanitized_histogram[sanitized_histogram<0] = 0
        return sanitized_histogram


    #################################################### attribute_binning #####################

    def dist_binning(self,eps,private_dist = False):
        domain_temp = copy.deepcopy(self.domain)
        data_temp = copy.deepcopy(self.data)
        # counts = np.zeros()
        self.ori_2_bin = dict()
        laplace_counts = dict()
        for attr in self.attr_list:
            attr_domainsize = self.domain.dict[attr]['domain']
            bin_num = self.bin_num
            if attr_domainsize > bin_num:
                logging.info(f'{self.ID} is binning the attribute {attr}')
                if self.config['binning_method'] == 'dist':
                    data_temp[:,attr - self.attr_list[0]] = pd.cut(list(data_temp[:,attr - self.attr_list[0]]), bin_num).codes
                    dic = dict()
                    for i in range(bin_num):
                        subset = self.data[data_temp[:,attr - self.attr_list[0]] == i]
                        target_values, target_counts = np.unique(subset[:, attr - self.attr_list[0]], return_counts=True)
                        if private_dist:
                            target_counts = self.sanitize_histogram(target_counts,self.budget_for_binning,1)
                            pass
                        else:
                            dic[i] =(target_values, target_counts/np.sum(target_counts))
                else:
                    eps = self.budget_for_binning
                    ori_2_bin, dic, counts = tools.dp_qcut(data_temp[:,attr - self.attr_list[0]],bin_num,eps)
                    # original_2_bin[attr] = ori_2_bin
                    laplace_counts[attr] = counts
                    temp = data_temp[:,attr - self.attr_list[0]]
                    for value in ori_2_bin:
                        temp[temp == value] =  ori_2_bin[value]
                    # data_temp[:,attr] = tools.bin_map(ori_2_bin,self.data[:,attr])
                    # data_temp[:,attr] = pd.qcut(list(data_temp[:,attr]), bin_num, duplicates='drop').codes
                self.bin_2_original[attr] = dic
            #     self.data[:,attr] = pd.cut(list(self.data[:,attr]), bin_num).codes
                domain_temp.dict[attr]['domain'] = bin_num
            else:
                dic = dict()
                data_temp_ = data_temp[:,attr - self.attr_list[0]]
                domain_size = domain_temp.dict[attr]['domain']
                for i in range(domain_size):
                    subset = data_temp_[data_temp_ == i]
                    target_values, target_counts = np.unique(subset, return_counts=True)
                    dic[i] =(target_values, target_counts/np.sum(target_counts))
                self.bin_2_original[attr] = dic
                _, counts = np.unique(data_temp[:,attr - self.attr_list[0]], return_counts=True)
                dim = len(counts)
                if self.budget_for_binning == 0:
                    counts = counts
                else:
                    noise = np.random.laplace(0, 1/self.budget_for_binning, dim)
                    counts = counts + noise
                counts[counts <= 0] = 2
                counts = [int(np.ceil(count)) for count in list(counts)]
                laplace_counts[attr] = counts
        return data_temp, domain_temp, self.bin_2_original, laplace_counts
    


    ############################################# LocMRF ###############################################
    
    def build_local_mrf(self):
        noisy_data_num = len(self.data)
        local_config = copy.copy(self.config)
        local_config['epsilon'] = np.round(self.budget_for_local_MRF*100)/100
        local_config['max_clique_size'] = 1e5
        local_config['global_clique_size'] =2e5
        local_config['max_parameter_size'] = 3e5
        local_config['init_measure'] = 0
        # if self.ID = 'Bob':
        attr_hierarchy = []
        for ind in self.attr_list:
            attr_hierarchy.append(self.attr_hierarchy[ind])
        self.attribute_graph = AttributeGraph(self.data, self.domain_client, attr_hierarchy, \
            local_config, self.config['data'])
        graph, measure_list, attr_hierarchy, attr_to_level, entropy, adj= self.attribute_graph.construct_model()
        mrf = MarkovRandomField(self.data, self.domain_client, graph, measure_list, attr_hierarchy, \
        attr_to_level, noisy_data_num, local_config, gpu=True)
        mrf.entropy_descent()
        return {'mrf':mrf, 'attr_list':self.attr_list, 'graph':graph, 'adj':adj}


    



    ############################################# upload message ###############################################
    def upload_msg(self):
        msg={}
        msg['ID'] = self.ID
        msg['attr_list'] = self.attr_list
        msg['attr_num'] = self.attr_num
        msg['n'] = self.data_num
        if self.config['private_method'] == 'fmsketch':
            msg['n'] = int(self.data_num + np.random.laplace(0, 1 / (self.budget_for_data_num)))

        # construct the local MRFS before attribute binning
        if self.config['local_MRF']:
            msg['mrf'] = self.build_local_mrf()
        else:
            msg['mrf'] = 0

        if self.config['attribute_binning']:
            # self.overall_optimal_binning()
            domain_change = dict()
            domain_ = copy.deepcopy(self.domain)
            self.data, domain_temp, self.bin_2_original, laplace_counts= self.dist_binning(self.budget_for_binning)
            for attr in self.attr_list:
                domain_change[attr] = (domain_.dict[attr]['domain'], domain_temp.dict[attr]['domain'])
            self.binning_map['data'] = self.data[:,tuple([attr-self.attr_list[0] for attr in self.attr_list])]
            self.binning_map['bin_2_original'] = self.bin_2_original
            self.binning_map['original_2_bin'] = {}
            self.binning_map['domain_change'] = domain_change
            msg['binning_map'] = self.binning_map
            msg['laplace_counts'] = laplace_counts
            self.domain = domain_temp
        
        if self.config['private_method'] == 'fmsketch':
            logging.info(f'{self.ID} is generating FMsketches under differential privacy!')
            self.generate_multiple_sketches()
            msg['budget_used'] = self.eps
        elif self.config['private_method'] == 'random_response':
            logging.info(f'{self.ID} is generating Random responses under differential privacy!')
            self.ldp_generate_random_responses()
            msg['budget_used'] = self.eps
        else:
            msg['budget_used'] = 0
        msg['private_statistics'] = self.private_statistics
        logging.info(f'##############################local end!')
        return msg
    

    
