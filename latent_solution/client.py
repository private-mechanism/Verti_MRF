import PrivMRF
import PrivMRF.utils.tools as tools
from PrivMRF.domain import Domain
from PrivMRF.fmsketch import complement_fm_sketch
from PrivMRF.utils import my_tools
from PrivMRF.utils.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from PrivMRF.attribute_graph import AttributeGraph
from PrivMRF.markov_random_field import MarkovRandomField

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
        self.attr_num = len(domain)
        self.gpu = gpu
        self.private_statistics = {}
        self.bin_cut ={}
        for attr in self.attr_list:
            self.bin_cut[attr] = self.config['binning_num']
        self.eps = eps
        self.bin_num = 2
        self.seeds = seeds
        self.binning_map = {}  # {attr:{domain:(21,2), cut:5},...}
        self.cut_map = {}
        self.attr_hierarchy = [attr_hierarchy[attr] for attr in self.attr_list]
        # for index, attr in enumerate(self.attr_list):
        #     self.attr_hierarchy.append(attr_hierarchy[attr])


    def clean_membership(self, attr):
        s = self.domain.dict[attr]['domain']
        clean_membership = []
        for i in range(s):
            clean_membership.append(set(np.where(self.data[:,attr] == i)[0]))
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
    
    #####################################################Generate private statistics based on fmsketch
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

    #####################################################Generate private statistics based on random response
    def one_hot_decoder(self,domain_size, value):
        code = np.zeros(domain_size)
        code[value] = 1
        return code
    
    def random_response(self,code):
        private_code = code.copy()
        eps = self.eps/16
        # for \epsilon < 1/4, it holds that random response is 8*epsilon-LDP
        assert eps < 1/4
        for i in range(code.shape[0]):
            r = random.uniform(0,1)
            if r < 1/2 - eps:
                private_code[i] = 1 - code[i]
        return code, private_code
    
    def generate_random_responses(self):
        for attr in self.attr_list:
            domain_size = self.domain.dict[attr]['domain']
            temp = np.zeros((self.data_num, domain_size))
            for k in range(self.data_num):
                value = self.data[k,attr]
                code = self.one_hot_decoder(domain_size, value)
                _, private_code = self.random_response(code)
                temp[k,:] =  private_code
            self.private_statistics[attr]= temp

    def ldp_generate_random_responses(self):
        for attr in self.attr_list:
            perturbed_value = self.ldp_perturb_attr(attr)
            self.private_statistics[attr] = perturbed_value


    # domain_size = self.domain.dict[attr]['domain']
    def ldp_perturb_attr(self, attr):
        domain_size = self.domain.dict[attr]['domain']
        real_values = self.data[:,attr]
        if domain_size > 3 * int(round(np.exp(self.eps))) + 2:
            # run OLH to perturb labels
            logging.info("===> using OLH for ldp")
            perturbed = volh_perturb(real_values, self.eps)
            # decode perturbation and get membership
            # perturbed_value = volh_membership(perturbed, domain= domain_size, g=int(round(np.exp(self.eps))) + 1)
        else:
            # run RR to perturb labels
            logging.info("===> using RR for ldp")
            perturbed = rr_perturb(real_values, self.eps, domain_size)
            # generate rr membership
            # perturbed_value = rr_membership(perturbed, domain_size)
            # todo: debug
            # print(f"*** rr true label histogram: {np.histogram(labels, bins=local_k, range=(0, domain_size))}")
            # print(f"*** rr perturbed label histogram: {np.histogram(perturbed, bins=local_k, range=(0, local_k))}")
        return perturbed
    


    def overall_optimal_binning(self):
        for attr in self.attr_list:
            logging.info(f'{self.ID} is binning the attribute {attr}')
            if self.domain.dict[attr]['domain'] > self.bin_num:
                self.single_optimal_binning(attr)
                self.data[:,attr] = self.cut_map[attr][1]
                self.binning_map[attr] = {}
                self.binning_map[attr]['domain_change'] = (self.domain.dict[attr]['domain'], self.bin_num)
                self.binning_map[attr]['cut'] = self.cut_map[attr][0]
                self.binning_map[attr]['data'] = self.cut_map[attr][1]
                self.domain.dict[attr]['domain'] = self.bin_num
            else:
                self.binning_map[attr] = {}
                self.binning_map[attr]['domain_change'] = (self.domain.dict[attr]['domain'], self.domain.dict[attr]['domain'])
                self.binning_map[attr]['cut'] = 0
                self.binning_map[attr]['data'] = self.data[:,attr]



    def single_optimal_binning(self, attr):
        # 暂时只适用于2分箱
        domain_size = self.domain.dict[attr]['domain']
        domain = copy.deepcopy(self.domain)
        domain.dict[attr]['domain'] = 2
        max_mi = 0
        opt_cut = tuple()
        for cut in range(domain_size-1):
            MI = 0
            data_temp = self.data.copy()
            temp = data_temp[:,attr].copy()
            temp[temp<=cut] = 0
            temp[temp>cut] = 1
            data_temp[:,attr] = temp
            for index in self.attr_list:
                if index != attr:
                    MI += self.mutual_info(data_temp, domain, (attr, index))
            if MI > max_mi:
                max_mi = MI
                opt_cut = cut, temp
        self.cut_map[attr] = opt_cut


    def mutual_info(self, data, domain, attr_pair):
        MI = self.entropy(data, domain, [attr_pair[0]])
        MI += self.entropy(data, domain, [attr_pair[1]])
        MI -= self.entropy(data, domain, list(attr_pair))
        return MI

    def entropy(self, data, domain, index_list):
        # data_attr_i = data[:,attr_pair[0]]
        # data_attr_j = data[:,attr_pair[1]]
        # domain_i = domain.dict[attr_pair[0]]['domain']
        # domain_j = domain.dict[attr_pair[0]]['domain']
        bins = domain.project(index_list).edge()
        # bins = temp_domain
        histogram, _= np.histogramdd(data[:, index_list], bins=bins)
        histogram = histogram.flatten()
        entropy = stats.entropy(histogram)
        return entropy


    # def binning(self):
    #     domain_temp = self.domian.copy()
    #     for attr in self.bin_cut:
    #         attr_domainsize = self.domain.dict[attr]['domain']
    #         bin_num = self.bin_cut[attr]
    #         if attr in self.attr_list and attr_domainsize > bin_num:
    #             logging.info(f'{self.ID} is binning the attribute {attr}')
    #             # if self.config['binning_method'] == 'dist':
    #                 # self.data[:,attr] = pd.cut(list(self.data[:,attr]), bin_num).codes
    #             # elif self.config['binning_method'] == 'freq':
    #             #     self.data[:,attr] = pd.qcut(list(self.data[:,attr]), bin_num).codes
    #             # else:
    #             self.data[:,attr] = pd.cut(list(self.data[:,attr]), bin_num).codes
    #             domain_temp.dict[attr]['domain'] = bin_num

    
    def build_local_mrf(self):
        noisy_data_num = len(self.data)
        self.attribute_graph = AttributeGraph(self.data, self.domain, self.attr_hierarchy, self.attr_list, self.config, self.config['data'])
        graph, measure_list, attr_hierarchy, attr_to_level, entropy= self.attribute_graph.construct_model()
        mrf = MarkovRandomField(self.data, self.domain, graph, measure_list, attr_hierarchy, \
        attr_to_level, noisy_data_num, self.config, gpu=True)
        mrf.entropy_descent()
        return mrf, self.attr_list


    def latent_model(self):
        # for attr in self.attr_list:
            
        return latent_data, condition_probability, group 


    def upload_msg(self):
        msg={}
        msg['ID'] = self.ID
        msg['attr_list'] = self.attr_list
        msg['attr_num'] = self.attr_num
        msg['n'] = int(self.data_num + np.random.laplace(0, 1 / (0.05 * self.eps)))
        if self.config['attribute_binning']:
            self.overall_optimal_binning()
            msg['binning_map'] = self.binning_map
        msg['mrf'] = self.build_local_mrf()
        if self.config['private_method'] == 'fmsketch':
            logging.info(f'{self.ID} is generating FMsketches under differential privacy!')
            self.eps *= 0.95
            # generate_multiple_sketches
            self.generate_multiple_sketches()
            #上传用于server生成需要被减掉的K_p
            msg['budget_used'] = self.eps
        elif self.config['private_method'] == 'random_response':
            logging.info(f'{self.ID} is generating Random responses under differential privacy!')
            self.ldp_generate_random_responses()
            msg['budget_used'] = self.eps/16
        msg['private_statistics'] = self.private_statistics
        return msg
    
    
    # def test(self, attr_pair):
    #     clean = self.intersection_clean(self,attr_pair)
    #     self.generate_multiple_sketches()
    #     fm = my_tools.fm_intersection_ca(self.config, attr_pair,self.domain, self.fmsketch, self.data_num, self.eps)

    