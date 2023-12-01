import PrivMRF
import PrivMRF.utils.tools as tools
from PrivMRF.domain import Domain
from PrivMRF.fmsketch import complement_fm_sketch
from PrivMRF.utils import my_tools
from PrivMRF.utils import corex
from PrivMRF.utils.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from PrivMRF.attribute_graph_local import AttributeGraph
from PrivMRF.markov_random_field import MarkovRandomField
from sklearn.preprocessing import OneHotEncoder
from tree import Latent_tree_model


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
            self.budget_for_local_MRF = self.eps*self.proportion_for_local_MRF/np.sqrt(data_parties)
            logging.info(f'the privbacy budget for generate local MRF for {self.ID} is {self.budget_for_local_MRF}')
            # self.eps

        self.budget_for_binning = 0
        if self.config['attribute_binning'] and (self.config['binning_method'] == 'freq' or not self.config['uniform_sampling']):
            '''
            the binning based on frequency can violent privacy concern.  
            '''
            self.proportion_for_binning = self.config['binning_theta']
            count = 0
            for attr in range(len(self.domain)):
                if self.domain.dict[attr]['domain'] > self.config['binning_num']:
                    count += 1
            if self.config['use_binning2consis']:
                self.budget_for_binning = self.eps*self.proportion_for_binning/np.sqrt(len(domain))
            else:
                self.budget_for_binning = self.eps*self.proportion_for_binning/np.sqrt(count)
        
        if self.config['private_method'] == 'fmsketch':
            self.proportion_for_data_num = self.config['data_num_theta']
            self.budget_for_data_num = self.eps*self.proportion_for_data_num
            self.eps = self.eps*(1-self.proportion_for_local_MRF-self.proportion_for_binning-self.proportion_for_data_num)/np.sqrt(len(self.domain))
        else:
            self.eps = self.eps*(1-self.proportion_for_local_MRF-self.proportion_for_binning)/np.sqrt(len(self.domain))
    
        if self.config['private_method'] == 'fmsketch':
            logging.info(f'budget for binning is {self.budget_for_binning}, budget for local MRF is {self.budget_for_local_MRF*np.sqrt(2)}, \
                budget for noisy data num{self.budget_for_data_num},for generating sketches is {self.eps*np.sqrt(len(self.domain))}')
        else:
            logging.info(f'budget for binning is {self.budget_for_binning}, budget for local MRF is {self.budget_for_local_MRF*np.sqrt(2)}, \
                budget for LDP is {self.eps*np.sqrt(len(self.domain))}')

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
    
    ################################################Generate private statistics based on fmsketch

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
        if self.ID == 'Bob':
            attr = attr - self.attr_list[0]
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

    # def single_dist_binning(self, attr):
    #     # 暂时只适用于2分箱
    #     domain_size = self.domain.dict[attr]['domain']
    #     domain = copy.deepcopy(self.domain)
        


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
    
    def sanitize_histogram(self,histogram,epsilon,sensitivity):
        dim = histogram.ndim
        noise = np.random.laplace(0, sensitivity/ epsilon,dim)
        sanitized_histogram = histogram + noise
        sanitized_histogram[sanitized_histogram<0] = 0
        return sanitized_histogram


    def dist_binning(self,eps,private_dist = False):
        domain_temp = copy.copy(self.domain)
        data_temp = copy.copy(self.data)
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
    


    ############################################# build local MRF ###############################################
    
    def build_local_mrf(self):
        noisy_data_num = len(self.data)
        local_config = copy.copy(self.config)
        local_config['epsilon'] = np.round(self.budget_for_local_MRF*100)/100
        local_config['max_clique_size'] = 100000
        local_config['global_clique_size'] = 100000
        local_config['init_measure'] = 1
        # if self.ID = 'Bob':
        attr_list = [i for i in range(len(self.attr_list))]
        new_dict = {key-self.attr_list[0]: self.domain.dict[key] for key in self.attr_list}
        new_attr_list = [attr for attr in attr_list]
        domain_client = Domain(new_dict,new_attr_list)
        attr_hierarchy = []
        for ind in self.attr_list:
            attr_hierarchy.append(self.attr_hierarchy[ind])
        self.attribute_graph = AttributeGraph(self.data, domain_client, attr_hierarchy, \
            local_config, self.config['data'])
        graph, measure_list, attr_hierarchy, attr_to_level, entropy, adj= self.attribute_graph.construct_model()
        mrf = MarkovRandomField(self.data, domain_client, graph, measure_list, attr_hierarchy, \
        attr_to_level, noisy_data_num, local_config, gpu=True)
        mrf.entropy_descent()
        return {'mrf':mrf, 'attr_list':self.attr_list, 'graph':graph, 'adj':adj}




    ############################################# build latent_model ###############################################

    def calculate_conditional_probability(self,latent_data, observe_data, target_attribute):
        attributes = self.latent_2_observe_dic[target_attribute]
        # 统计目标属性的取值及其计数
        target_values, target_counts = np.unique(latent_data[:, target_attribute], return_counts=True)
        # 创建一个字典来存储条件概率
        conditional_probabilities = {}
        # 遍历目标属性的取值
        for target_value, target_count in zip(target_values, target_counts):
            # 在目标属性等于给定取值的情况下，统计各属性的取值及其计数
            subset = observe_data[latent_data[:, target_attribute] == target_value]
            attribute_values = []
            attribute_counts = []
            for attribute in attributes:
                values, counts = np.unique(subset[:, attribute], return_counts=True)
                attribute_values.append(values)
                attribute_counts.append(counts)
            # 计算条件概率
            conditional_prob = {}
            for combination in np.ndindex(*[len(values) for values in attribute_values]):
                probability = 1.0
                for attribute, value_index in zip(attributes, combination):
                    index = np.where(attributes == attribute)[0][0]
                    probability *= attribute_counts[index][value_index] / target_count
                conditional_prob[tuple(combination)] = probability
            # 将条件概率存储到字典中
            conditional_probabilities[target_value] = conditional_prob
        return conditional_probabilities
    


    def build_latent_model(self):
        X = self.data
        self.config['latent_num'] = 4
        self.config['latent_dimension'] = 4
        latent_factor_num = self.config['latent_num'] 
        latent_factor_dim = self.config['latent_dimension'] 
        Latent = corex.Corex(n_hidden=latent_factor_num, dim_hidden=latent_factor_dim)  
        # Define the number of hidden factors to use.
        Latent.fit(X)
        group = Latent.clusters
        self.latent_2_observe_dic = {}
        self.observe_2_latent_dic = {}
        self.latent_2_observe_prob = {}
        self.target_2_choice = {}
        for latent_factor in range(latent_factor_num):
            attrs = np.where(group == latent_factor)[0]
            self.latent_2_observe_dic[latent_factor] = attrs
            for attr in attrs:
                self.observe_2_latent_dic[attr] = latent_factor
        # for attr in self.attr_list:
        latent_data = Latent.labels
        for i in range(latent_factor_num):
            self.latent_2_observe_prob[i] = self.calculate_conditional_probability(latent_data, self.data, i)
        return latent_data, self.latent_2_observe_dic, self.latent_2_observe_prob
    

    def syn_local_data(self, syn_latent_data):
        # 根据每个attr的条件概率生成数据
        attr_list = [str(attr) for attr in self.attr_list]
        syn_data = pd.DataFrame(columns = attr_list)
        local_latent_num = self.config['latent_num']
        latent_factor_dim = self.config['latent_dimension']
        dic = {}
        for attr in attr_list:
            dic[attr] = []
        for row in range(syn_latent_data.shape[0]):
            for latent_factor in range(local_latent_num):
                value = syn_latent_data[row][latent_factor]
                attrs = self.latent_2_observe_dic[latent_factor]
                attrs = [str(attr) for attr in attrs]
                tuples = list(self.latent_2_observe_prob[latent_factor][value].keys())
                probs = list(self.latent_2_observe_prob[latent_factor][value].values())
                tuple_idx = np.random.choice(range(len(tuples)), p = probs)
                Attrs_data = np.array(tuples[tuple_idx])
                # new_df = pd.DataFrame(columns = attrs)
                for i, attr in enumerate(attrs):
                    val = Attrs_data[i]
                    dic[attr].append(val)
        # new_df = new_df.assign(**dic)
        syn_data = syn_data.assign(**dic)
        data = syn_data.values

        tools.write_csv(list(data), list(range(self.attr_num)), './preprocess/' + 'syn.csv')
        # pd.concat([syn_data,new_df],axis = 0,ignore_index=True)
                # syn_data = syn_data.append(new_df, ignore_index=True)
                # pd.concat
        return data



    # evaluate dp data on k way marginal task
    def k_way_marginal(self, data_name, dp_data_list, k, marginal_num):
        # data, headings = utils.tools.read_csv('./exp_data/' + data_name + '_train.csv')
        data, headings = tools.read_csv('./preprocess/' + data_name + '.csv', print_info=False)
        data = np.array(data, dtype=int)
        data = data[:,tuple(self.attr_list)]
        encoder = OneHotEncoder(sparse=False)
        data = encoder.fit_transform(data)

        attr_num = data.shape[1]
        data_num = data.shape[0]
        # domain = json.load(open('./preprocess/'+data_name+'.json'))
        # domain = {int(key): domain[key] for key in domain}
        # domain = Domain(domain, list(range(attr_num)))

        domain = {int(attr): {"type": "discrete", "domain": 2} for attr in list(range(attr_num))}
        domain = Domain(domain, list(range(attr_num)))


        marginal_list = [tuple(sorted(list(np.random.choice(attr_num, k, replace=False)))) for i in range(marginal_num)]
        marginal_dict = {}
        size_limit = 1e8
        for marginal in marginal_list:
            temp_domain = domain.project(marginal)
            if temp_domain.size() < size_limit:
                # It is fast when domain is small, howerver it will allocate very large array
                edge = temp_domain.edge()
                histogram, _ = np.histogramdd(data[:, marginal], bins=edge)
                marginal_dict[marginal] = histogram
            else:
                uniques, cnts = np.unique(data, return_counts=True, axis=0)
                uniques = [tuple(item) for item in uniques]
                cnts = [int(item) for item in cnts]
                marginal_dict[marginal] = dict(zip(uniques, cnts))

        # total variation distance
        tvd_list = []
        # for dp_data_path in dp_data_list:
        for dp_data in dp_data_list:
            # dp_data, headings = utils.tools.read_csv(dp_data_path, print_info=False)
            dp_data = np.array(dp_data, dtype=int)
            dp_data_num = len(dp_data)
            tvd = 0
            # print('data:', dp_data_path)
            for marginal in marginal_dict:
                temp_domain = domain.project(marginal)
                if temp_domain.size() < size_limit:
                    edge = temp_domain.edge()
                    histogram, _ = np.histogramdd(dp_data[:, marginal], bins=edge)
                    histogram *= data_num/dp_data_num
                    temp_tvd = np.sum(np.abs(marginal_dict[marginal] - histogram)) / len(data) / 2
                else:
                    uniques, cnts = np.unique(dp_data, return_counts=True, axis=0)
                    uniques = [tuple(item) for item in uniques]
                    cnts = [int(item)*data_num/dp_data_num for item in cnts]
                    diff = []
                    unique_cnt = marginal_dict[marginal]
                    for i in range(len(uniques)):
                        if uniques[i] in unique_cnt:
                            diff.append(cnts[i] - unique_cnt[uniques[i]])
                        else:
                            diff.append(cnts[i])
                    diff = np.array(diff)
                    # TVD = 1/2 * sum(abs(diff)) = 1.0 * sum(max(diff, 0))
                    diff[diff<0] = 0
                    temp_tvd = np.sum(diff)/len(data)

                if temp_tvd > 1:
                    print(marginal, temp_domain.size(), temp_tvd)
                tvd += temp_tvd
                # print('    {}: {}'.format(marginal, temp_tvd))
            tvd /= len(marginal_dict)
            tvd_list.append(tvd)
        return tvd_list
    
    def test(self):
        latent_data, latent_2_observe_dic, latent_2_observe_prob = self.build_latent_model()
        syn_data = self.syn_local_data(latent_data)
        encoder = OneHotEncoder(sparse=False)
        syn_data = encoder.fit_transform(syn_data)
        tvd_list = self.k_way_marginal('adult', [syn_data], 3, 200)
        return tvd_list


    def latent_model_construction(self):
        group_size = self.config['group_size']
        epsilon = self.config['epsilon']*1/2
        LTM = Latent_tree_model(self.data, self.domain, self.attr_list, self.config, group_size,epsilon,private=False)
        G, Y, pyg_dict,py_dict = LTM.TLAG(self.data, self.domain, self.attr_list, group_size, epsilon)
        pg_dict = {}
        for latent_attr in Y:
            pg_dict[latent_attr] = {}
            pattern_dict,pattern_array = LTM.pattern_fre(self.data, G[latent_attr])
            value_list = [[0,1] for attr in G[latent_attr]]
            for value in itertools.product(*value_list):
                pg_dict[latent_attr][value] = pattern_dict[value]/np.sum(pattern_array)
                # {frequency/np.sum(pattern_array) for pattern,frequency in pattern_dict}
        if self.ID == 'Bob':
            latent_start_index = np.ceil((len(self.domain)-len(self.attr_list))/group_size)
            Y = [int(latent_attr + latent_start_index) for latent_attr in Y]
            pyg_dict_ = {}
            py_dict_ = {}
            pg_dict_ = {}
            for key in pyg_dict.keys():
                pyg_dict_[int(key+latent_start_index)] = pyg_dict[key]
                py_dict_[int(key+latent_start_index)] = py_dict[key]
                pg_dict_[int(key+latent_start_index)] = pg_dict[key]
            return G, Y, pyg_dict_,py_dict_, pg_dict_
        return G, Y, pyg_dict,py_dict, pg_dict
    



    ############################################# upload message ###############################################
    def upload_msg(self):
        msg={}
        msg['ID'] = self.ID
        msg['attr_list'] = self.attr_list
        msg['attr_num'] = self.attr_num
        msg['n'] = self.data_num
        if self.config['private_method'] == 'fmsketch':
            msg['n'] = int(self.data_num + np.random.laplace(0, 1 / (self.budget_for_data_num)))

        

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

        # construct the local MRFS before attribute binning
        if self.config['local_MRF']:
            msg['mrf'] = self.build_local_mrf()
        else:
            msg['mrf'] = 0
        
        
        if self.config['private_method'] == 'fmsketch':
            logging.info(f'{self.ID} is generating FMsketches under differential privacy!')
            self.generate_multiple_sketches()
            #上传用于server生成需要被减掉的K_p
            msg['budget_used'] = self.eps
        elif self.config['private_method'] == 'random_response':
            logging.info(f'{self.ID} is generating Random responses under differential privacy!')
            self.ldp_generate_random_responses()
            msg['budget_used'] = self.eps
        elif self.config['private_method'] == 'scalar_product':
            logging.info(f'{self.ID} is scalar_product!')
            msg['budget_used'] = 0
        elif self.config['private_method'] == 'latent_mrf':
            # logging.info(f'{self.ID} is generating the latent attribute!')
            G, Y, pyg_dict,py_dict,pg_dict = self.latent_model_construction()
            self.private_statistics['G'] = G
            self.private_statistics['Y'] = Y
            self.private_statistics['pyg_dict'] = pyg_dict
            self.private_statistics['py_dict'] = py_dict
            self.private_statistics['pg_dict'] = pg_dict
            msg['budget_used'] = self.eps
            logging.info(f'{self.ID} is generating the latent attributes {G}!')
        else:
            msg['budget_used'] = 0
        msg['private_statistics'] = self.private_statistics
        return msg
    

    