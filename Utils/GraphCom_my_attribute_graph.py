from .utils import tools
import networkx as nx
import numpy as np
import random
import itertools
import math
from .domain import Domain
from networkx.readwrite import json_graph
import json
import pickle
from .factor import Factor
import time
import logging
import copy

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

class AttributeGraph:
    def __init__(self, data, bin_domain, domain, noisy_data_num, attr_list, attr_index, config, data_name, fmsketches,private_statistics, eps):
        self.data = data.copy()
        self.bin_domain = bin_domain
        self.domain = domain
        self.attr_list = attr_list
        self.attr_index = attr_index
        self.config = config
        self.data_name = data_name

        self.attr_num = len(attr_index)
        privacy_budget = eps*np.sqrt(len(domain))
        self.privacy_budget = config['beta1'] * tools.privacy_budget(privacy_budget)
        self.fmsketches = fmsketches

        self.entropy_map = {}
        self.TVD_map = {} 
        self.MI_map = {}
        self.private_statistics = private_statistics
        # if self.config['private_method'] == 'fmsketch':
        #     self.fmsketches = private_statistics
        # elif self.config['private_method'] == 'random_response':
        #     self.random_responses = private_statistics

        self.eps = eps    # DP LEVEL FOR EACH ATTRIBUTE USED FOR DENOSING!!!
        self.noisy_data_num = noisy_data_num
        self.gpu = True

        if data_name == 'acs':
            self.config['size_penalty'] = 1e-7
            self.config['max_clique_size'] = 2e5
            self.config['max_parameter_size'] = 1e6

        if data_name == 'adult' and self.config['epsilon'] < 0.41:
            self.config['max_parameter_size'] = 3e7

        data_num_noise = math.sqrt(1/self.privacy_budget*0.01)
        self.data_num = int(len(self.data) + np.random.normal(scale=data_num_noise))

        # self.config['max_entropy_num'] = (self.attr_num ** 2)*2
        self.config['max_entropy_num'] = (self.attr_num ** 2)/2 + self.attr_num
        self.config['entropy_num_iter'] = self.attr_num*self.attr_num/2
        self.config['search_iter_num'] = int(self.attr_num*(self.attr_num-1))


        # optimal Gaussian Mehchanism
        # self.max_entropy_num = self.config['max_entropy_num']
        # sensitivity = tools.entropy_sensitivity(self.data_num)
        # self.entropy_noise = math.sqrt(self.max_entropy_num * sensitivity * sensitivity / self.privacy_budget)
        # entropy is for debugging and not used to generate synthetic data
        self.entropy_noise = 0

        max_edge_num = self.attr_num * (self.attr_num - 1)/2
        self.privacy_budget_per_TVD = self.privacy_budget/np.sqrt((max_edge_num+len(domain)))
        self.noisy_histogram_attr ={}
        for attr in range(len(domain)):
            temp_domain = domain.project([attr])
            histogram, _= np.histogramdd(self.data[:, (attr,)], bins=temp_domain.edge())
            self.noisy_histogram_attr[attr] = histogram + np.random.laplace(0, 1/self.privacy_budget_per_TVD, histogram.shape)

        # sensitivity = tools.MI_sensitivity(self.data_num)
        # self.MI_noise = math.sqrt(max_edge_num * sensitivity * sensitivity / self.privacy_budget)

        sensitivity = tools.TVD_sensitivity(self.data_num)
        self.TVD_noise = math.sqrt(max_edge_num * sensitivity * sensitivity / self.privacy_budget)

        self.max_measure_dom_2way = 0
        self.max_measure_dom_high_way = 0
        if self.config['init_measure'] == 3:
            self.config['init_measure_num'] = 0
        else:
            self.config['init_measure_num'] = self.attr_num
        
        estimated_noise_scale = 'nan'
        if self.config['beta2'] > 0:
            budget = (1-self.config['beta1']-self.config['beta3']) * tools.privacy_budget(self.config['epsilon'])
            estimated_noise_scale = math.sqrt((self.attr_num + self.config['t']*self.attr_num) / budget)
            self.max_measure_dom_2way = self.data_num / estimated_noise_scale / config['theta1']
            self.max_measure_dom_high_way = self.data_num / estimated_noise_scale / config['theta2']

        self.min_score = -1e8

        self.max_level = [max(self.attr_list[attr].level_to_size.keys()) for attr in range(self.attr_num)]

        # the best measure for each attr to avoid worst cases
        self.attr_measure = {}
        
        print('privacy budget:           ', self.privacy_budget)
        print('estimated noise scale:    ', estimated_noise_scale)
        print('max 2way measure dom:     ', self.max_measure_dom_2way)
        print('max high way measure dom: ', self.max_measure_dom_high_way)
        print('max edge number:          ', max_edge_num)
        print('TVD_noise:                ', self.TVD_noise)
        print('data num:                 ', self.data_num)

    def construct_model(self, mrf_msg):
        print('construct attribute graph')
        if not self.config['combine_MRF']:
            self.graph, entropy = self.local_search()
        else:
            self.graph, entropy, adj = self.attr_graph_combine(mrf_msg)
        # self.graph, entropy = self.pairwise_graph()

        self.attr_to_level = None
        if self.config['enable_attribute_hierarchy']:
            self.attr_to_level = {i: max(self.attr_list[i].level_to_size.keys()) for i in range(self.attr_num)}
            for attr in range(self.attr_num):
                if self.attr_to_level[attr] > 0:
                    print('attr: {} max_level: {}'.format(attr, self.attr_to_level[attr]))
            if self.config['init_measure'] != 3:
                if self.config['epsilon'] < 0.41 and self.data_name == 'adult':
                    if self.config['epsilon'] <= 0.21:
                        self.attr_to_level[3] = 0
                        self.attr_to_level[13] = 1
                    elif self.config['epsilon'] <= 0.41:
                        self.attr_to_level[3] = 1
                        self.attr_to_level[13] = 1

        self.measure_list = []

        if not self.config['combine_MRF']:
            if self.config['init_measure'] == 0:
                measure_list = self.construct_inner_Bayesian_network()
            elif self.config['init_measure'] == 1:
                measure_list = self.get_all_n_way_measure(2)
            elif self.config['init_measure'] == 2:
                measure_list = list(tuple(sorted(clique)) for clique in nx.find_cliques(self.graph))
            elif self.config['init_measure'] == 3:
                self.measure_list = []
                return self.graph, self.measure_list, self.attr_list, self.attr_to_level, entropy
            elif self.config['init_measure'] == 4:
                # self.measure_list = []
                return self.graph, self.measure_list, self.attr_list, self.attr_to_level, entropy
            else:
                print('error: invaild init_measure')
                exit(-1)

            # add most valuable measures for attrs
            self.measure_list = []
            for attr in self.attr_measure:
                self.measure_list.append(self.attr_measure[attr][0])
            print('attr measure',  self.measure_list)
            self.measure_list = \
                list(tools.deduplicate_measure_set(tuple(sorted(measure)) \
                    for measure in self.measure_list))
        else:
            alice_measure_list = list(mrf_msg['alice']['mrf'].measure_set)
            correct = len(mrf_msg['alice']['attr_list'])
            for measure in mrf_msg['bob']['mrf'].measure_set:
                lis = []
                for attr in list(measure):
                    lis.append(attr + correct)
                alice_measure_list.append(tuple(lis))
            self.measure_list = alice_measure_list

        # determine the level of attribute hierarchy
        if self.config['enable_attribute_hierarchy']:
            for measure in self.measure_list:
                if len(measure) == 2:
                    # print(measure, tools.measure_level_size(measure, self.attr_list, self.attr_to_level), self.max_measure_dom_2way)
                    for i in range(3):
                        if tools.measure_level_size(measure, self.attr_list, self.attr_to_level) > self.max_measure_dom_2way:
                            tools.improve_level(measure, self.attr_list, self.attr_to_level, self.config['max_level_gap'])
            
            for attr in self.attr_to_level:
                if self.max_level[attr] > 0:
                    print('  attr: {}, level: {} max_level: {}'.format(attr, self.attr_to_level[attr], self.max_level[attr]))

        attr_flag = [0] * self.attr_num
        for measure in self.measure_list:
            for attr in measure:
                attr_flag[attr] += 1
        for attr in range(self.attr_num):
            if attr_flag[attr] == 0: 
                self.measure_list.append(tuple([attr]))
                print('single attr measure:', attr)

        # run acs on cpu, so we have to use a small graph
        # note this step actually decrease the performance
        if self.data_name == 'acs' and self.config['epsilon'] > 0.10:
            self.graph = nx.Graph()
            self.graph.add_nodes_from(list(range(self.attr_num)))
            for measure in self.measure_list:
                for attr1, attr2 in itertools.combinations(measure, 2):
                    self.graph.add_edge(attr1, attr2)

        data = json_graph.node_link_data(self.graph)
        with open('./temp/graph_'+self.config['exp_name']+'.json', 'w') as out_file:
            json.dump(data, out_file)
        return self.graph, self.measure_list, self.attr_list, self.attr_to_level, entropy

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as out_file:
            pickle.dump(model, out_file)

    @staticmethod
    def load_model(path, config=None):
        with open(path, 'rb') as out_file:
            model = pickle.load(out_file)
        if config != None:
            model.config = config
        return model


    def get_all_n_way_measure(self, n):
        measure_list = []
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for clique in self.maximal_cliques:
            for measure in itertools.combinations(clique, n):
                measure_list.append(measure)
        return measure_list
    
    def get_all_cross_silo_measure(self,n):
        measure_list = []
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for clique in self.maximal_cliques:
            for measure in itertools.combinations(clique, n):
                measure_list.append(measure)
        return measure_list


    # randomly enumerate a next edge to find a graph that minimize KL divergence and get measures
    def local_search(self):
        # with open('./temp/graph_'+self.config['exp_name']+'.json', 'r') as in_file:
        #     graph = json_graph.node_link_graph(json.load(in_file))
        #     return graph, -1
        start_G = nx.Graph()
        start_G.add_nodes_from(list(range(self.attr_num)))
        start_adj = np.zeros(shape=(self.attr_num, self.attr_num), dtype=float)
        # data_entropy = tools.dp_entropy({}, self.data, self.bin_domain, list(range(self.attr_num)), 0)[0]
        # print('data entropy: {}'.format(data_entropy))
        start_status = Status(start_G, start_adj)
        current_status = start_status
    

        score_func = self.pairwise_score_TVD
        local_count = 0
        search_iter_num = self.config['search_iter_num']
        entropy = -1
        # check_entropy_map = {}

        for i in range(search_iter_num):
            # generate edge list
            best_score = self.min_score
            best_status = None
            edge_list = []
            for attr1 in range(self.attr_num):
                for attr2 in range(attr1+1, self.attr_num):
                    if current_status.adj[attr1][attr2] == 0:
                        edge_list.append((attr1, attr2))
            random.shuffle(edge_list)

            for attr1, attr2 in edge_list:
                status = current_status.get_neighbor_status(attr1, attr2, 1)
                status_score, mutual_info, size = score_func(status.graph)
                # print('status score:', status_score)
                if status_score > best_score:
                    best_score = status_score
                    best_status = status
                    best_mutual_info = mutual_info
                    current_size = size

            if best_status == None:
                print('  found local minimum')
                local_count += 1
                if local_count >= 3:
                    break
                continue
            # local_count = 0
            current_status = best_status
            current_score = best_score
            current_mutual_info = best_mutual_info
            # print entropy for debug, which could be very slow as it need to calculate the entropy of
            # large marginals
            # _, entropy, _ = self.score(current_status.graph, check_entropy_map)
            entropy = -1
            print('  iter: {}/{} score: {:.2f} size: {:.2e}, edge_num: {} mutual_info: {:.2f}'\
                .format(i, search_iter_num, current_score, current_size, \
                current_status.graph.number_of_edges(), current_mutual_info))
        graph = current_status.graph
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        tools.print_graph(graph, './temp/graph_'+self.config['exp_name']+'.png')
        return graph, entropy



    def construct_inner_Bayesian_network(self):
        self.R_noise = None
        # add budget for constructing inner Bayesian network
        print('construct Bayesian Network for maximal cliques')
        measure_list = []
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for i in range(len(self.maximal_cliques)):
            clique = self.maximal_cliques[i]
            print('  {}, {}/{}'.format(clique, i, len(self.maximal_cliques)))
            measure_list.extend(self.greedy_Bayes(clique))
        
        if self.config['supplement_2way']:
            for clique in self.maximal_cliques:
                for edge in itertools.combinations(clique, 2):
                    measure_list.append(edge)

        measure_list = list(set(measure_list))
        return measure_list
    


    def maximal_parents(self, parents_set, dom):
        if dom < 1:
            return set()
        if len(parents_set) < 1:
            return set([tuple(),])
        # print(parents_set, dom)
        parents_set = parents_set.copy()
        attr = parents_set.pop()
        res1 = self.maximal_parents(parents_set, dom)

        # If there exists a high level subattr satisfying dom limitation
        # It should be considered as levels don't influence the scores of parents
        # debug
        if self.config['enable_attribute_hierarchy']:
            level = max(self.max_level[attr]-2, 0)
        else:
            level = max(self.max_level[attr], 0)
            # level = 2
        
        current_attr_size = self.attr_list[attr].level_to_size[level]
        # current_attr_size = 2
        res2 = self.maximal_parents(parents_set, dom/current_attr_size)
        for ps in res2:
            if ps in res1:
                res1.remove(ps)
            temp = list(ps)
            temp.append(attr)
            res1.add(tuple(sorted(temp)))

        return res1
    
    def select_parents(self, remaining_attributes, parents_set):
        attr_parents = []
        for attr in remaining_attributes:
            dom_limit = self.max_measure_dom_high_way/self.bin_domain.project([attr]).size()
            attr_parents.extend([(attr, parent) for parent in self.maximal_parents(parents_set, dom_limit)])

        best_attr = None
        best_parent = []
        best_score = self.min_score
        for ap in attr_parents:
            score = self.attr_parents_score(ap[0], list(ap[1]))
            if score > best_score:
                best_score = score
                best_attr = ap[0]
                best_parent = ap[1]

        if best_attr == None:
            best_attr = random.choice(remaining_attributes)

        marginal = [best_attr]
        marginal.extend(best_parent)
        marginal = tuple(sorted(marginal))

        return (marginal, best_attr), best_score

    # use greedy bayes to find measures
    def greedy_Bayes(self, clique):
        remaining_attributes = list(clique.copy())
        best_attr = random.choice(remaining_attributes)

        remaining_attributes.remove(best_attr)
        parents_set = [best_attr]

        measure_list = []
        while len(remaining_attributes) != 0:
            marginal_item, score = self.select_parents(remaining_attributes, parents_set)
            best_attr = marginal_item[1]
            best_parents = marginal_item[0]

            if len(marginal_item[0]) > 1:
                print('    {}<={} score: {}'.format(best_attr, best_parents, score))
                best_parents = tuple(sorted(best_parents))
                measure_list.append(best_parents)

                if best_attr not in self.attr_measure or self.attr_measure[best_attr][1] < score:
                    self.attr_measure[best_attr] = [best_parents, score]

                remaining_attributes.remove(best_attr)
                parents_set.append(best_attr)
            else:
                print('unable to construct Bayes network under dom limitation', clique)
                break
        
        return measure_list
    

    # TVD correlation + correlation-based feature selector
    # ref: Correlation-based Feature Selection for Machine Learning
    def attr_parents_score(self, attr, parents):
        parents = list(parents).copy()
        # If the measure constructed by the only parents of one attribute is too large,
        # it should be add to the model regardless of its size as it provide basic correlation we can get
        # It will aslo be used to determine attribute hierarchy if possible
        # however, if a measure is too large, the noise will also be very large.
        # It will even affect the entire model
        if len(parents) == 1:
            dom_limit = self.max_measure_dom_2way
            if self.config['enable_attribute_hierarchy']:
                for pa in parents:
                    if self.attr_to_level[pa] > 0:
                        dom_limit = self.max_measure_dom_2way * 5
                if self.attr_to_level[attr] > 0:
                    dom_limit = self.max_measure_dom_2way * 5
        else:
            dom_limit = self.max_measure_dom_high_way
        if self.bin_domain.project(parents).size() * self.bin_domain.dict[attr]['domain'] > dom_limit:
            return self.min_score
        numerator = 0
        for i in parents:
            # it will reuse the TVD queried before
            if self.config['score'] == 'pairwsie_TVD':
                # numerator += tools.dp_TVD(self.TVD_map, self.data, self.bin_domain, [attr, i], self.TVD_noise)[1]
                numerator += self.dp_TVD([attr, i])
            elif self.config['score'] == 'pairwsie_MI':
                numerator += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.bin_domain, [attr, i], self.MI_noise)[1]
            else:
                print('score must be pairwsie_TVD or pairwsie_MI')
                exit(-1)
        denominator = len(parents)
        for i in range(len(parents)):
            for j in range(i+1, len(parents)):
                if self.config['score'] == 'pairwsie_TVD':
                    denominator += tools.dp_TVD(self.TVD_map, self.data, self.bin_domain, [i, j], self.TVD_noise)[1]
                elif self.config['score'] == 'pairwsie_MI':
                    numerator += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.bin_domain, [i, j], self.MI_noise)[1]
        # denominator might be smaller than 0 because of noise. We set it at least 1 as len(parents) >= 1
        if denominator < 1:
            denominator = 1
        return numerator/math.sqrt(denominator)


    def pairwise_score_MI(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.bin_domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        noisy_MI = 0
        for attr1, attr2 in graph.edges:
            noisy_MI += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.bin_domain, [attr1, attr2], self.MI_noise)[1]
        
        score = noisy_MI - self.config['size_penalty']*size
        return score, noisy_MI, size
    
    def pairwise_score_TVD(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        noisy_TVD = 0
        for attr1, attr2 in graph.edges:
            # noisy_TVD += tools.dp_TVD(self.TVD_map, self.data, self.bin_domain, [attr1, attr2], self.TVD_noise)[1]
            noisy_TVD += self.dp_TVD([attr1, attr2])
        
        score = noisy_TVD - self.config['size_penalty']*size
        return score, noisy_TVD, size

    def pairwise_score(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        noisy_mutual_info = 0
        mutual_info = 0
        for attr1, attr2 in graph.edges:
            entropy, noisy_entropy  = tools.dp_entropy(self.entropy_map, self.data, self.bin_domain, [attr1, attr2], self.entropy_noise)
            mutual_info             -= entropy
            noisy_mutual_info       -= noisy_entropy

            entropy, noisy_entropy  = tools.dp_entropy(self.entropy_map, self.data, self.bin_domain, [attr1], self.entropy_noise)
            mutual_info             += entropy
            noisy_mutual_info       += noisy_entropy

            entropy, noisy_entropy  = tools.dp_entropy(self.entropy_map, self.data, self.bin_domain, [attr2], self.entropy_noise)
            mutual_info             += entropy
            noisy_mutual_info       += noisy_entropy

        score = noisy_mutual_info - self.config['size_penalty']*size
        return score, mutual_info, size

    def score(self, graph, entropy_map):
        graph = graph.copy()
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        clique_list = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(clique_list)
        for clique1, clique2 in itertools.combinations(clique_list, 2):
            clique_graph.add_edge(clique1, clique2, weight=-len(set(clique1) & set(clique2)))
        junction_tree = nx.minimum_spanning_tree(clique_graph)
        # print('    clique list', len(clique_list), clique_list)

        # junction tree size
        size = 0
        for clique in clique_list:
            temp_size = self.bin_domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        # KL divergence
        # model entropy is for debugging and can not be used for constructing model as they are not dp
        KL_divergence = 0
        model_entropy = 0
        entropy, noisy_entropy = tools.dp_entropy(entropy_map, self.data, self.bin_domain, clique_list[0], self.entropy_noise)
        KL_divergence += noisy_entropy
        model_entropy += entropy
        for start, clique in nx.dfs_edges(junction_tree, source=clique_list[0]):
            entropy, noisy_entropy = tools.dp_entropy(entropy_map, self.data, self.bin_domain, clique, self.entropy_noise)
            KL_divergence += noisy_entropy
            model_entropy += entropy
            separator = set(start) & set(clique)
            if len(separator) != 0:
                entropy, noisy_entropy = tools.dp_entropy(entropy_map, self.data, self.bin_domain, separator, self.entropy_noise)
                KL_divergence -= noisy_entropy
                model_entropy -= entropy

        # print('KL', KL_divergence, size)
        score = -KL_divergence - self.config['size_penalty']*size

        return score, model_entropy, size
    
    def check_clique_size(self,graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return False
            size += temp_size
        if size > self.config['max_parameter_size']:
            return False
        return True

    
    
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
                histogram = self.ldp_intersection_ca(index_list)
                fact1 = Factor(domain, histogram, np)
                temp_domain = domain.project([index_list[0]])
                histogram= self.ldp_intersection_ca((index_list[0],))    
                fact2 = Factor(temp_domain, histogram, np)
                temp_domain = domain.project([index_list[1]])
                histogram= self.ldp_intersection_ca((index_list[1],))  
                fact3 = Factor(temp_domain, histogram, np)
            fact4 = fact2.expand(domain) * fact3.expand(domain) / self.noisy_data_num
            TVD = np.sum(np.abs(fact4.values - fact1.values)) / 2 / self.noisy_data_num
            if self.gpu:
                TVD = TVD.item()
        return TVD
    
    
    

    def attr_graph_combine(self, mrf_msg):
        # graph_msg = [(mrf_msg['alice']['mrf'].graph, mrf_msg[0][1], mrf_msg[0][2]),(mrf_msg[1][0].graph, mrf_msg[1][1],mrf_msg[1][2])]
        # first initialize two local attr graph
        graph_1 = mrf_msg['alice']['graph']
        attr_list_1 = mrf_msg['alice']['attr_list']
        adj_1 = mrf_msg['alice']['adj']
        graph_2 = mrf_msg['bob']['graph']
        attr_list_2 = mrf_msg['bob']['attr_list']
        adj_2 = mrf_msg['bob']['adj']

        # initialize the global attr graph based on the two local graphs
        initial_graph = nx.disjoint_union(graph_1, graph_2)
        initial_adj = np.zeros(shape=(len(self.bin_domain), len(self.bin_domain)), dtype=float)
        n1, d1 = adj_1.shape
        n2, d2 = adj_2.shape
        initial_adj[0:n1, 0:d1] = adj_1
        initial_adj[n1:(n1+n2), d1:(d1+d2)] = adj_2

        start_status = Status(initial_graph, initial_adj)
        current_status = start_status
        entropy = -1
        max_count = len(attr_list_1)*len(attr_list_2)
        self.config['search_iter_num'] = max_count 

        max_count = len(attr_list_1)*len(attr_list_2)
        tvd_dic = {}
        for attr1 in attr_list_1:
            for attr2 in attr_list_2:
                tvd_dic[(attr1, attr2)]=self.dp_TVD([attr1, attr2])

        max_index = int(np.ceil(max_count/5))
        tvd_dic = sorted(tvd_dic.items(), key=lambda item:item[1], reverse=True)
        edge_list = []
        for edge, TVD in tvd_dic:
            edge_list.append(edge)
        count = 0
        status = copy.deepcopy(current_status)
        for attr1, attr2 in edge_list:
            status = status.get_neighbor_status(attr1, attr2, 1)
            if self.check_clique_size(status.graph):
                current_status = status
                count += 1
            else:
                status = copy.deepcopy(current_status)
            if count == max_index:
                break
        graph = current_status.graph
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        tools.print_graph(graph, './temp/graph_'+self.config['exp_name']+'.png')
        return graph, entropy, current_status.adj
    


    #####################################################fmsketch-based CarEst
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
            all_sketches[idx] =  self.one_round_intersection_alpha(index_list, idx)
        debias = 0.7213 / (1 + 1.079 / m)
     
        # epsilon, delta = priv_config['eps'], priv_config['delta']
        domain_size = self.bin_domain.dict[index_list[0]]['domain']
        c = len(index_list)*(domain_size-1)
        # len(splits) * (len(splits[0]) - 1)
        k_p, _ = self.set_k_p_min(self.eps, 1/self.noisy_data_num, m, gamma)
        # the offset (k_p) may need to be revised, because here we are doing the complementary
        raw_comlementary_union = m / np.sum(np.power(1 + gamma, -all_sketches), axis=0) * debias - k_p * c
        # print(k_p, len(splits))
        # print(raw_comlementary_union)
        estimate = self.noisy_data_num - raw_comlementary_union
        estimate[estimate<0] = 10
        estimate = estimate * self.noisy_data_num/np.sum(estimate)
        histogram = self.clean_intersection_ca(index_list)
        shape = tuple([self.bin_domain.dict[index_list[i]]['domain'] for i in range(len(index_list))])
        DP_FM_histogram = estimate.reshape(shape)
        # print(f"DP estimate: {estimate}")
        return DP_FM_histogram
    

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
        one_way_sketches = []
        for idx in range(m):
            one_way_sketches.append(self.fmsketches[idx][attr]['private_statistics'])
        debias = 0.7213 / (1 + 1.079 / m)
        k_p, _ = self.set_k_p_min(self.eps, 1/self.noisy_data_num, m, gamma)
        complementary_estimate = m / np.sum(np.power(1 + gamma, -np.array(one_way_sketches)), axis=0) * debias - c*k_p
        estimate = self.noisy_data_num - complementary_estimate
        estimate[estimate< 0] = 0
        estimate  = estimate * self.noisy_data_num/np.sum(estimate)
        clean_estimate = self.clean_histogram_ca(attr)
            # one_ways.append(raw_estimate)
        return estimate


    ################################################################## FO-based CarEst
    def rr_histogram_ca(self, rr_data, index_list):
        da = rr_data
        tu = index_list
        temp_domain = self.bin_domain.project(index_list)
        histogram, _= np.histogramdd(rr_data[:, index_list], bins=temp_domain.edge())
        return histogram
    

    def multiplyList(self,myList):
        result = 1
        for x in myList:
            result = result * x  
        return result
    
    def cartesian_to_index(self, combine, domain_list):
        idx = 0
        total = len(combine)
        for i, e in enumerate(combine):
            temp = self.multiplyList(domain_list[i+1:])
            idx += e * temp
        return idx
    

    def index_to_cartesian(self, idx, domain_list):
        tmp = idx
        catesian = [0] * len(domain_list)
        i = len(domain_list) - 1
        while tmp > 0:
            tmp, mod = divmod(tmp, domain_list[i])
            catesian[i] = mod
            i -= 1
        return catesian
    

    def ldp_intersection_ca(self, index_list):
        # index_list = self.attr_list
        # print(self.data[0,:])
        intersection = self.rr_histogram_ca(self.data, index_list)
        from functools import partial
        def flatten(x):
            original_shape = x.shape
            return x.flatten(), partial(np.reshape, newshape=original_shape)
        adjusted, unflatten = flatten(intersection)
        all_combines = [list(range(self.bin_domain.dict[attr]['domain'])) for attr in index_list]
        all_combines = list(itertools.product(*all_combines))
        #todo: adapting to the cases where domain sizes of attrs are different
        domain_list = [self.bin_domain.dict[index_list[i]]['domain'] for i in range(len(index_list))]
        # intersection_num = np.power(domain_size, len(index_list))
        intersection_num = self.multiplyList(domain_list)
        eps = self.eps
        p_list = []
        q_list = []
        for domain_size in domain_list:
            if domain_size > 3 * int(round(np.exp(eps))) + 2:
                g = int(round(np.exp(eps))) + 1
                p = np.exp(eps) / (np.exp(eps) + g - 1)
                q = 1.0 / (np.exp(eps) + g - 1)
            else:
                p = np.exp(eps) / (np.exp(eps) + domain_size - 1)
                q = 1.0 / (np.exp(eps) + domain_size - 1)
            p_list.append(p)
            q_list.append(q)

        # generate forward probability matrix
        
        # max_idx = self.multiplyList(domain_list)
        forward_probs = np.ones(shape=(intersection_num, intersection_num))
        for combine in all_combines:
            idx1 = self.cartesian_to_index(combine, domain_list)
            for idx2 in range(idx1, intersection_num):
                inner_combine = self.index_to_cartesian(idx2, domain_list)
                for i in range(len(combine)):
                    if inner_combine[i] == combine[i]:
                        forward_probs[idx1, idx2] *= p_list[i]
                        # forward_probs[idx2, idx1] *= p_list[i]
                    else:
                        forward_probs[idx1, idx2] *= q_list[i]
                        # forward_probs[idx2, idx1] *= q_list[i]
                forward_probs[idx2, idx1] = forward_probs[idx1, idx2]
                # diff = np.count_nonzero(np.array(combine) != np.array(inner_combine))
                # forward_probs[idx1, idx2] = np.power(q, diff) * np.power(p, len(index_list) - diff)
                # forward_probs[idx2, idx1] = np.power(q, diff) * np.power(p, len(index_list)- diff)
        # compute unbiased frequencies
        inv_prob = np.linalg.inv(forward_probs)
        # todo: debug
        # print(f"******* sizes: {inv_prob.shape}, {adjusted.shape}")
        # histogram = self.clean_intersection_ca(index_list)
        adjusted = np.matmul(inv_prob, adjusted)
        # logging.info(f"sum of adjust {np.sum(adjusted)}")
        adjusted[adjusted < 0] = 0
        adjusted = adjusted/np.sum(adjusted)*len(self.data)

        # self.private_statistics = unflatten(adjusted)
        return unflatten(adjusted)
    

