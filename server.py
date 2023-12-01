# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg

# thread number for numpy (when it runs on CPU)
import numpy as np
import os
from functools import reduce
import math
import numpy as np
import networkx as nx
import itertools
from scipy.optimize import fsolve
import numpy as np
import itertools
import logging
import copy
import random
import pickle

from PrivMRF.preprocess import preprocess
from PrivMRF.factor import Factor
from PrivMRF.my_markov_random_field import MarkovRandomField
from PrivMRF.fmsketch import intersection_ca
from PrivMRF.domain import Smoother,Domain
from PrivMRF.my_attribute_graph import AttributeGraph
from PrivMRF.utils.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from PrivMRF.preprocess import read_preprocessed_data, postprocess
from PrivMRF.attribute_hierarchy import get_one_level_hierarchy
from PrivMRF.utils import my_tools

thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num


class Status:
    def __init__(self, graph, adj):
        self.graph = graph.copy()
        self.adj = np.copy(adj)

    def get_neighbor_status(self, attr1, attr2, weight):
        neighbor_status = Status(self.graph, self.adj)
        if neighbor_status.adj[attr1, attr2] > 0:
            neighbor_status.adj[attr1, attr2] = 0
            neighbor_status.adj[attr2, attr1] = 0
            neighbor_status.graph.remove_edge(attr1, attr2)
            return neighbor_status
        elif neighbor_status.adj[attr1, attr2] == 0:
            neighbor_status.adj[attr1, attr2] = weight
            neighbor_status.adj[attr2, attr1] = weight
            neighbor_status.graph.add_edge(attr1, attr2)
            return neighbor_status
        return None



class Server:
    def __init__(self, data, attr_list, seeds, attr_hierarchy,\
    domain, config, gpu=True):
        self.attr_list = attr_list
        # self.attr_to_level = attr_to_level
        self.config = config
        self.domain = domain
        self.bin_domain = copy.deepcopy(domain)
        if self.config['structure_entropy']:
            self.noisy_data_num = self.data_num
        self.max_measure_attr_num = config['max_measure_attr_num']
        # self.attr_num = len(domain)
        self.gpu = gpu
        self.fmsketches = {}
        self.random_responses = {}
        self.noisy_data_num = 0
        self.TVD_map = {}
        self.eps = 0
        self.seeds = seeds
        self.data =data
        self.raw_data = copy.deepcopy(data)
        self.attr_hierarchy = attr_hierarchy
        self.rr_intersection_histogram_dict={}
        self.mrf = None
        self.marginal_set = []
        self.candidate_marginal_set = None


    def recieve_msg(self, msg_list):
        name = self.config['data']+self.config['private_method']
        with open('result/inner_res/'+name+'.pkl', 'wb+') as f:  # Python 3: open(..., 'wb')
            pickle.dump(msg_list, f)
        n = 0
        epsilon = 0
        temp_msg=[]
        mrf_msg = []
        self.mrf_msg = {}
        data_list = []
        self.binning_map = {}
        bin_2_original = {}
        original_2_bin = {}
        domain_change = {}
        laplace_counts = {}


        
        for msg in msg_list:
            n += msg['n']
            epsilon += msg['budget_used']
            temp_msg.append(msg['private_statistics'])
            mrf_msg.append(msg['mrf'])
            
            if self.config['attribute_binning']:
                data_list.append(msg['binning_map']['data'])
                bin_2_original.update(msg['binning_map']['bin_2_original'])
                domain_change.update(msg['binning_map']['domain_change'])
                laplace_counts.update(msg['laplace_counts'])
                original_2_bin.update(msg['binning_map']['original_2_bin'])
        self.mrf_msg['alice'] = mrf_msg[0]
        self.mrf_msg['bob'] = mrf_msg[1]
                

        if self.config['private_method'] == 'fmsketch':
            for idx in range(self.config['m']):
                temp_msg[0][idx].update(temp_msg[1][idx])
                self.fmsketches[idx] = temp_msg[0][idx].copy()
        elif self.config['private_method'] == 'latent_mrf':
            self.Y = []
            self.G = []
            self.pyg_dict = {}
            self.py_dict = {}
            self.pg_dict = {}
            for ele in temp_msg:
                self.Y.extend(ele['Y'])
                self.G.extend(ele['G'])
                self.pyg_dict.update(ele['pyg_dict'])
                self.py_dict.update(ele['py_dict'])
                self.pg_dict.update(ele['pg_dict'])
        else:
            temp_msg[0].update(temp_msg[1])
            self.random_responses = temp_msg[0].copy()
        self.noisy_data_num = n/len(msg_list)
        # self.eps = epsilon/len(msg_list)
        self.eps = epsilon/len(msg_list)

        if self.config['private_method'] == 'random_response':
            print(self.data[0,:])
            self.data = self.ldp_recover_data()
            print(self.data[0,:])


        if self.config['attribute_binning']:
            for attr in self.attr_list:
                if self.bin_domain.dict[attr]['domain'] > self.config['binning_num']:
                    self.bin_domain.dict[attr]['domain'] = self.config['binning_num'] 
            if self.config['private_method'] == 'random_response':
                self.data = self.ldp_recover_data()
            else:
                self.data= np.hstack((data_list[0],data_list[1]))
            self.binning_map['bin_2_original'] = bin_2_original
            self.binning_map['original_2_bin'] = original_2_bin
            self.binning_map['domain_change'] = domain_change
            self.binning_map['laplace_counts'] = laplace_counts
            # self.binning_map['data'] = self.data
            
        


    #####################################################fmsketch-based counting
    def set_k_p_min(self, epsilon, delta, m, gamma):
        """A helper function for computing k_p and eta."""
        if not 0 < epsilon < float('inf') or not 0 < delta < 1:
            k_p = 0
            alpha_min = 0
        else:
            eps1 = epsilon / 4 / np.sqrt(m * np.log(1 / delta))
            k_p = np.ceil(1 / (np.exp(eps1) - 1))
            alpha_min = np.ceil(-np.log(1 - np.exp(-eps1)) / np.log(1 + gamma))
        return k_p, alpha_min


    def one_round_intersection_alpha(self, index_list, idx):
        sketch = []
        for i in range(len(index_list)):
            sketch.append(self.fmsketches[idx][index_list[i]]['private_statistics'])
        cartesian = list(itertools.product(*sketch))
        return [np.max(c) for c in cartesian]

    
    def fm_intersection_ca(self, index_list):
        m = self.config['m']
        gamma = self.config['gamma']
        num_intersections = np.product([self.bin_domain.dict[index_list[i]]['domain'] for i in range(len(index_list))])
        all_sketches = np.zeros(shape=(m, num_intersections))
        for idx in range(m):
            all_sketches[idx] = self.one_round_intersection_alpha(index_list, idx)
        debias = 0.7213 / (1 + 1.079 / m)
        # epsilon, delta = priv_config['eps'], priv_config['delta']
        domain_size = self.bin_domain.dict[index_list[0]]['domain']
        c = len(index_list)*(domain_size-1)
        # len(splits) * (len(splits[0]) - 1)
        k_p, _ = self.set_k_p_min(self.eps, 1/self.noisy_data_num, m, gamma)
        # the offset (k_p) may need to be revised, because here we are doing the complementary
        raw_comlementary_union = m / np.sum(np.power(1 + gamma, -all_sketches), axis=0) * debias - k_p * c
        estimate = self.noisy_data_num - raw_comlementary_union
        estimate[estimate<0] = 10
        estimate = estimate * self.noisy_data_num/np.sum(estimate)
        histogram = self.clean_intersection_ca(index_list)
        shape = tuple([self.bin_domain.dict[index_list[i]]['domain'] for i in range(len(index_list))])
        DP_FM_histogram = estimate.reshape(shape)
        loss_1_norm = np.sum(np.abs(DP_FM_histogram-histogram))
        # print(f"DP estimate: {estimate}")
        return DP_FM_histogram, loss_1_norm
    
    def test_seed(self):
        loss = 0
        for attr_1 in self.attr_list:
            for attr_2 in self.attr_list:
                if attr_2>attr_1:
                    if self.config['private_method'] == 'fmsketch':
                        _,loss_temp = self.fm_intersection_ca((attr_1,attr_2))
                    else:
                        _,loss_temp = self.ldp_intersection_ca((attr_1,attr_2))
                    loss += loss_temp
        return loss
        
    

    def clean_intersection_ca(self, attr_pair):
        domain = self.bin_domain.project(attr_pair)
        bins = domain.edge()
        histogram, _= np.histogramdd(self.data[:, attr_pair], bins=bins)
        return histogram
    

    def clean_histogram_ca(self, attr):
        temp_domain = self.bin_domain.project([attr])
        temp_index_list = temp_domain.attr_list
        histogram, _= np.histogramdd(self.data[:, temp_index_list], bins=temp_domain.edge())
        return histogram


    def fm_histogram_ca(self, attr):
        m = self.config['m']
        gamma = self.config['gamma']
        c = self.bin_domain.dict[attr]['domain']-1
        # domain_size = self.bin_domain.dict[attr]['domain']
        one_way_sketches = []
        for idx in range(m):
            one_way_sketches.append(self.fmsketches[idx][attr]['private_statistics'])
        # estimate one party's ca
        debias = 0.7213 / (1 + 1.079 / m)
        # one_ways = []
        # if self.priv_config:
            # epsilon, delta = priv_config['eps'], priv_config['delta']
        k_p, _ = self.set_k_p_min(self.eps, 1/self.noisy_data_num, m, gamma)
        # else:
        #     k_p = 0
        # for one_way_sketch in all_one_way_sketches:
        complementary_estimate = m / np.sum(np.power(1 + gamma, -np.array(one_way_sketches)), axis=0) * debias - c*k_p
        estimate = self.noisy_data_num - complementary_estimate
        estimate[estimate< 0] = 0
        estimate  = estimate * self.noisy_data_num/np.sum(estimate)
        clean_estimate = self.clean_histogram_ca(attr)
            # one_ways.append(raw_estimate)
        return estimate
    
    #####################################################rr-based counting

    def ldp_recover_data(self):
        self.intersection_dic = {}
        self.histogram_dic = {}
        data_num = len(self.random_responses[self.attr_list[0]])
        perturbed_data = np.zeros([data_num, len(self.attr_list)])
        for attr in self.attr_list:
            perturbed_data[:,attr] = self.random_responses[attr]
        return perturbed_data
    

    # def rr_generate_private_statistics(self):
    #     intersection_matrix = self.ldp_intersection_ca(self.data, self.attr_list)
    #     for attr1 in self.attr_list:
    #         for attr2 in self.attr_list:
    #             if attr2 > attr1:
    #                 temp_list = range(attr1)+range(attr1+1, attr2)+ range(attr2+1, len(self.attr_list))
    #                 self.intersection_dic[(attr1, attr2)]= np.sum(intersection_matrix,axis=tuple(temp_list))
    #     for attr in self.attr_list:
    #         temp_list = range(attr1)+range(attr1+1, attr2)+ range(attr2+1, len(self.attr_list))
    #         self.intersection_dic[(attr1, attr2)]= np.sum(intersection_matrix,axis=tuple(temp_list))
    #     for attr in self.attr_list:
    #         temp_list = range(attr)+range(attr+1, len(self.attr_list))
    #         self.histogram_dic[attr] = np.sum(intersection_matrix,axis=tuple(temp_list))
    #     return {'intersection':self.intersection_dic,'intersection_matrix':intersection_matrix, 'histogram':self.histogram_dic}
    
    

    def rr_histogram_ca(self, rr_data, index_list):
        temp_domain = self.bin_domain.project(index_list)
        histogram, _= np.histogramdd(rr_data[:, index_list], bins=temp_domain.edge())
        return histogram
    
    def cartesian_to_index(self, combine, local_k):
        idx = 0
        total = len(combine)
        for i, e in enumerate(combine):
            idx += e * np.power(local_k, total - i - 1)
        return idx

    def index_to_cartesian(self, idx, k, parties):
        tmp = idx
        catesian = [0] * parties
        i = parties - 1
        while tmp > 0:
            tmp, mod = divmod(tmp, k)
            catesian[i] = mod
            i -= 1
        return catesian
    

    def ldp_intersection_ca(self, index_list):
        # index_list = self.attr_list
        intersection = self.rr_histogram_ca(self.data, index_list)
        from functools import partial
        def flatten(x):
            original_shape = x.shape
            return x.flatten(), partial(np.reshape, newshape=original_shape)
        adjusted, unflatten = flatten(intersection)
        # adjusted = np.array(intersection_counts)
        all_combines = [list(range(self.bin_domain.dict[attr]['domain'])) for attr in index_list]
        all_combines = list(itertools.product(*all_combines))
        #todo: adapting to the cases where domain sizes of attrs are different
        domain_size = self.bin_domain.dict[index_list[0]]['domain']
        intersection_num = np.power(domain_size, len(index_list))
        eps = self.eps
        if domain_size > 3 * int(round(np.exp(eps))) + 2:
            g = int(round(np.exp(eps))) + 1
            p = np.exp(eps) / (np.exp(eps) + g - 1)
            q = 1.0 / (np.exp(eps) + g - 1)
        else:
            p = np.exp(eps) / (np.exp(eps) + domain_size - 1)
            q = 1.0 / (np.exp(eps) + domain_size - 1)

        # generate forward probability matrix
        forward_probs = np.ones(shape=(intersection_num, intersection_num)) * np.power(q, len(index_list))
        for combine in all_combines:
            idx1 = self.cartesian_to_index(combine, domain_size)
            for idx2 in range(idx1, np.power(domain_size, len(index_list))):
                inner_combine = self.index_to_cartesian(idx2, domain_size, len(index_list))
                diff = np.count_nonzero(np.array(combine) != np.array(inner_combine))
                forward_probs[idx1, idx2] = np.power(q, diff) * np.power(p, len(index_list) - diff)
                forward_probs[idx2, idx1] = np.power(q, diff) * np.power(p, len(index_list)- diff)
        # compute unbiased frequencies
        inv_prob = np.linalg.inv(forward_probs)
        # todo: debug
        # print(f"******* sizes: {inv_prob.shape}, {adjusted.shape}")
        adjusted = np.matmul(inv_prob, adjusted)
        # logging.info(f"sum of adjust {np.sum(adjusted)}")
        adjusted[adjusted < 0] = 0
        histogram = self.clean_intersection_ca(index_list)
        DP_rr_histogram = unflatten(adjusted)
        loss_1_norm = np.sum(np.abs(DP_rr_histogram-histogram))
        # self.private_statistics = unflatten(adjusted)
        return DP_rr_histogram, loss_1_norm
    

    

    # def rr_intersection_ca(self, attr_pair):
    #     domain_size_0 = self.bin_domain.dict[attr_pair[0]]['domain']
    #     domain_size_1 = self.bin_domain.dict[attr_pair[1]]['domain']
    #     n, d = self.random_responses[attr_pair[0]].shape
    #     rr_intersection_histogram = np.zeros((domain_size_0,domain_size_1))
    #     for i in range(domain_size_0):
    #         for j in range(domain_size_1):
    #             vector_0 = (self.random_responses[attr_pair[0]][:,i]-1/2+self.eps)/(2*self.eps)
    #             vector_1 = (self.random_responses[attr_pair[1]][:,j]-1/2+self.eps)/(2*self.eps)
    #             temp = np.inner(vector_0, vector_1)
    #             rr_intersection_histogram[i,j] = temp
    #     rr_intersection_histogram[rr_intersection_histogram<0] = 0
    #     return rr_intersection_histogram * n/np.sum(rr_intersection_histogram)
    

    # def rr_intersection_dic(self):
    #     attr_list_temp = self.attr_list.copy()
    #     for attr1 in attr_list_temp:
    #         for attr2 in attr_list_temp:
    #             if attr2 > attr1:
    #                 self.rr_intersection_histogram_dict[(attr1,attr2)] = self.rr_intersection_ca([attr1,attr2])

    
    # def rr_histogram_ca_(self, attr):
    #     temp_dict = self.rr_intersection_dic.copy()
    #     domain_size = self.bin_domain.dict[attr]['domain']
    #     n, d = self.random_responses[attr].shape
    #     num_other_attr = len(self.attr_list)-1
    #     raw_estimate = np.zeros(domain_size)
    #     # np.zeros((domain_size, num_other_attr))
    #     for key in temp_dict.keys():
    #         if attr in key:
    #             raw_estimate += temp_dict[key].sum(axis = 1)
    #     clean_estimate = self.clean_histogram_ca(attr)
    #     estimate = raw_estimate/num_other_attr
    #     estimate = estimate* n/np.sum(estimate)
    #         # one_ways.append(raw_estimate)
    #     return estimate
    
    ########################################### compute the noisy R-scores based

    def dp_TVD(self, index_list):
        domain = self.bin_domain
        TVD_map = {}
        if not isinstance(index_list, tuple):
            index_list = tuple(sorted(index_list))
        if index_list not in TVD_map:
            domain = domain.project(index_list)
            if self.config['private_method'] == 'fmsketch':
                histogram = self.fm_intersection_ca(index_list)
                fact1 = Factor(domain, histogram, np)
                temp_domain = domain.project([index_list[0]])
                histogram= self.fm_histogram_ca(index_list[0])
                fact2 = Factor(temp_domain, histogram, np)
                temp_domain = domain.project([index_list[1]])
                histogram= self.fm_histogram_ca(index_list[1])
                fact3 = Factor(temp_domain, histogram, np)
            else:
                histogram = self.rr_intersection_histogram_dict[(index_list[0],index_list[1])]
                fact1 = Factor(domain, histogram, np)
                temp_domain = domain.project([index_list[0]])
                histogram= self.rr_histogram_ca(self.data,index_list[0])
                fact2 = Factor(temp_domain, histogram, np)
                temp_domain = domain.project([index_list[1]])
                histogram= self.rr_histogram_ca(self.data,index_list[1])
                fact3 = Factor(temp_domain, histogram, np)
            fact4 = fact2.expand(domain) * fact3.expand(domain) / self.noisy_data_num
            TVD = np.sum(np.abs(fact4.values - fact1.values)) / 2 / self.noisy_data_num
            if self.gpu:
                TVD = TVD.item()
        return TVD
    
    def generate_intersection_sta(self):
        max_measure_attr_num = self.config['max_measure_attr_num']
        # for n in range(1, max_measure_attr_num+1):
        for n in range(1, 2):
            for measure in itertools.combinations(self.attr_list, n):
                self.intersection_dic[tuple(measure)]= self.fm_intersection_ca(measure)

    def generate_histogram_sta(self):
        self.histogram_dic = {}
        for attr in self.attr_list:
            self.histogram_dic[attr] = self.fm_histogram_ca(attr)


    def fm_generate_private_statistics(self):
        self.intersection_dic = {}
        # if self.config['binary'] == False:
        self.generate_intersection_sta()
        self.generate_histogram_sta()
        return {'intersection':self.intersection_dic,'histogram':self.histogram_dic}
    
    def lr_generate_private_statistics(self):
        private_statistics = {}
        private_statistics['Y']= self.Y
        private_statistics['G']= self.G
        private_statistics['pyg_dict'] = self.pyg_dict
        private_statistics['pg_dict'] = self.pg_dict
        private_statistics['py_dict'] = self.py_dict
        return private_statistics
    
    
    def build_attribute_graph(self,private_statistics):
        '''
        Initialize an attribute graph by comparing the R-scores
        '''
        if self.config['private_method'] == 'latent_mrf':
            self.attr_list = [i for i in range(len(self.Y))]
            temp = {}
            for i in self.attr_list:
                temp[i] = {"type": "discrete", "domain": 2}
            self.bindomain = Domain(temp, self.attr_list)                    
            self.attribute_graph = AttributeGraph(self.data, self.bin_domain, self.domain, self.noisy_data_num, self.attr_hierarchy,\
                                                self.attr_list, self.config, self.config['data'],self.fmsketches,private_statistics, self.eps)
            graph, measure_list, attr_hierarchy, attr_to_level, entropy= self.attribute_graph.construct_model(self.mrf_msg)
        else:
            self.attribute_graph = AttributeGraph(self.data, self.bin_domain, self.domain, self.noisy_data_num, self.attr_hierarchy,\
                                                self.attr_list, self.config, self.config['data'],self.fmsketches,private_statistics, self.eps)
            graph, measure_list, attr_hierarchy, attr_to_level, entropy= self.attribute_graph.construct_model(self.mrf_msg)
        return graph, measure_list, attr_hierarchy, attr_to_level, entropy
    


    ############################################# construct the markov random field
    def construct_mrf(self,graph, measure_list, attr_hierarchy, attr_to_level,private_statistics):
        # private_statistics = {}
        # if self.config['private_method'] == 'fmsketch':
        #     private_statistics = self.fmsketches
        # else:
        #     private_statistics = self.random_responses
        if self.config['private_method'] == 'fmsketch':
            mrf = MarkovRandomField(self.data, self.raw_data,self.bin_domain, self.domain, graph, measure_list, attr_hierarchy, self.attr_list,
            attr_to_level, self.noisy_data_num, self.config, private_statistics, self.fmsketches, self.eps, self.mrf_msg, self.binning_map,gpu=True)
        elif self.config['private_method'] == 'random_response':
            mrf = MarkovRandomField(self.data, self.raw_data,self.bin_domain, self.domain, graph, measure_list, attr_hierarchy, self.attr_list,
            attr_to_level, self.noisy_data_num, self.config, private_statistics, {}, self.eps, self.mrf_msg,self.binning_map,gpu=True)
        elif self.config['private_method'] == 'latent_mrf':
            mrf = MarkovRandomField(self.data, self.raw_data,self.bin_domain, self.domain, graph, measure_list, attr_hierarchy, self.attr_list,
            attr_to_level, self.noisy_data_num, self.config, private_statistics, {}, self.eps, self.mrf_msg,self.binning_map,gpu=True)
        else:
            mrf = MarkovRandomField(self.data, self.raw_data,self.bin_domain, self.domain, graph, measure_list, attr_hierarchy, self.attr_list,
            attr_to_level, self.noisy_data_num, self.config, private_statistics, self.fmsketches, self.eps, self.mrf_msg, self.binning_map,gpu=True)
        return mrf


    def build_global_mrf(self,private_statistics):
        graph, measure_list, attr_hierarchy, attr_to_level, entropy= self.build_attribute_graph(private_statistics)
        self.mrf = self.construct_mrf(graph, measure_list, attr_hierarchy, attr_to_level,private_statistics)
        self.mrf.entropy_descent()
        return self.mrf


    
    
    def candidate_marginal_selection(self):
        '''
        Generate the candidate marginal set based on the $\theta$-usefullness metric.
        '''
        self.candidate_marginal_set = self.mrf.generate_measure_set()
        self.max_measure_dom_2way = self.mrf.max_measure_dom_2way
        self.max_measure_dom_high_way = self.mrf.max_measure_dom_high_way
        self.min_score = self.attribute_graph.min_score


    def refine_marginal_set(self,initialized_marginal_set):
        consider_measure_list = self.candidate_marginal_set
        self.mrf.measure_set = initialized_marginal_set
        self.mrf.entropy_descent(consider_measure_list)
        return self.mrf.measure_set
    

    def generate_data(self,path):
        synthetic_data = self.mrf.synthetic_data(path)
        return synthetic_data
    

    # def evaluation():
    #     return metrics
    # sensitivity of TVD
    # ref: PrivBayes: Private Data Release via Bayesian Networks


    def TVD_sensitivity(n):
        return 2.0/n

    def triangulate(graph):
        edges = set()
        G = nx.Graph(graph)
        nodes = sorted(graph.degree(), key=lambda x: x[1])
        for node, degree in nodes:
            local_complete_edges = set(itertools.combinations(G.neighbors(node), 2))
            edges |= local_complete_edges
            G.add_edges_from(local_complete_edges)
            G.remove_node(node)
        triangulated_graph = nx.Graph(graph)
        triangulated_graph.add_edges_from(edges)
        return triangulated_graph

    
    





