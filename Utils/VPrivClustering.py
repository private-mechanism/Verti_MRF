import numpy as np
# import copy
from sklearn.cluster import KMeans
import itertools
import logging
import copy

from .VBase import VBase
from .solver_factory import solver_mapping
from util.save_results import save_result_to_json
from util.eval_centers import eval_centers, eval_homogeneity_score
from util.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
from util.load_config import generate_local_config
from util.fmsketch import intersection_ca
from util.postprocess import norm_sub
from util.local_k import local_k_choose

'''
This is an implementation of the paper Hu Ding et al. "K-Means Clustering with Distributed Dimensions"
The paper introduces a non-private VFL solution for K-means clustering problem
'''


class VPrivClustering(VBase):
    def __init__(self, config, tag, **kwargs):
        super().__init__(config, tag, )
        self.config = config
        self.tag = tag
        self.k = config['k']
        assert 'eps' in config and 'intersection_method' in config
        self.eps = config['eps']
        self.intersection_method = config['intersection_method']

        if 'local_solver' in config:
            self.local_solver = solver_mapping[config['local_solver']]
        else:
            self.local_solver = solver_mapping['basic']
        self.centers = None
        self.private_intersections = []
        self.clean_intersections = []
        self.clean_centers = None

    def fit(self, data, run_clean: bool = True, true_labels: np.array = None):
        if self.intersection_method in ['fmsketch']:
            n = int(self.config['n'] + np.random.laplace(0, 1 / (0.05 * self.eps)))
            self.eps *= 0.95
        else:
            n = self.config['n']
        centers = []
        clean_memberships = []
        parties = len(data)
        # current version use half of the eps for finding local centroids, half for reporting membership
        local_eps = self.eps / 2 / parties

        if 'local_k' in self.config:
            if self.config['local_k'] == 'auto':
                delta = 1 / data[0].shape[0]
                # todo: local eps is fixed to equal to local clustering eps
                local_k = local_k_choose(max_k=10, n=data[0].shape[0],
                                         number_sketches=self.config['m'],
                                         epsilon=local_eps, delta=delta)
                logging.info(f"auto set local k as {local_k}")
            elif self.config['local_k'] < np.power(self.config['k'], 1 / parties):
                local_k = int(np.ceil(np.power(self.config['k'], 1 / parties)))
            else:
                local_k = self.config['local_k']
        else:
            local_k = self.config['k']

        local_config = generate_local_config(self.config['d'], self.config['n'], local_k, eps=local_eps)
        for idx, subset_data in enumerate(data):
            # each party local run k-means and get centers
            logging.info(f"--> Working on client {idx} {subset_data.shape}")
            solver = self.local_solver(local_config, self.tag)
            solver.fit(subset_data)
            # -> get centers
            centers.append(list(solver.cluster_centers_))
        logging.info("local kmeans finished...")

        # if self.local_solver == solver_mapping['basic']:
        #     # for experiments: if the non-private k-means algorithm is applied on each party's local data
        #     # to show the effect of different intersection algorithms
        #     local_eps *= 2
        logging.info(f"Privacy budget for computing aggregation: {local_eps}")

        clean_memberships = []
        for idx, subset_data in enumerate(data):
            membership = self.clean_membership(subset_data, centers[idx])
            clean_memberships.append(membership)

        grids, intersection_counts, clean_intersection_counts = self.build_weighted_grids(n,
                                                                                          data,
                                                                                          centers,
                                                                                          clean_memberships,
                                                                                          local_eps,
                                                                                          local_k,
                                                                                          run_clean)

        logging.info(f"# of grid nodes: {len(grids)}; # of intersections: {len(intersection_counts)}")
        logging.info(f"intersection sizes: {intersection_counts}")

        if 'normalize' in self.config and self.config['normalize']:
            logging.info(f"postprocessing...")
            intersection_counts = norm_sub(intersection_counts, n=n)

        if self.intersection_method == 'random':
            chosen_idxs = np.random.choice(len(grids), size=self.k).astype(int)
            self.centers = np.array(grids)[chosen_idxs]
            loss = eval_centers(data, self.centers)
        else:
            # run k-means again on the weighted centers
            logging.info(f"central server runs k-means on grids with weights, k={self.k}")
            final_solver = KMeans(n_clusters=self.k, random_state=0)
            # print(grids)
            # print(intersection_counts)
            final_solver.fit(grids, sample_weight=np.array(intersection_counts) + 1e-5)

            loss = eval_centers(data, final_solver.cluster_centers_)
            self.centers = final_solver.cluster_centers_

        self.private_intersections = intersection_counts
        losses = {"private_final_loss": loss}

        if run_clean:
            # run k-means again on the weighted centers
            logging.info("running with non-private intersection for comparison...")
            clean_final_solver = KMeans(n_clusters=self.k, random_state=0)
            clean_final_solver.fit(grids, sample_weight=np.array(clean_intersection_counts) + 1e-5)
            clean_score = eval_centers(data, clean_final_solver.cluster_centers_)
            losses["clean_final_loss"] = clean_score
            self.clean_centers = clean_final_solver.cluster_centers_
            logging.info(f"intersection diff {intersection_counts - clean_intersection_counts}")
            if 'label_score' in self.config:
                # print(self.config['label_score'], self.config['label_score'] == True)
                losses['homogeneity'], losses['completeness'] = eval_homogeneity_score(data, self.centers, true_labels)

        self.save_results(losses)
        return self.centers

    def build_weighted_grids(self, n, data, centers, clean_memberships, eps, local_k, run_clean):
        memberships = []
        parties = self.config['T']
        # cartesian product of the local centers
        logging.info("generate cartesian product of local centers ...")
        cartesian = list(itertools.product(*centers))
        grids = []
        for combine in cartesian:
            grids.append(np.array(list(combine)).flatten())

        if self.intersection_method in ['noisymin', 'ldp']:
            # generate private memberships with noisymin or grr/olh
            for idx, subset_data in enumerate(data):
                # get randomized membership
                if self.intersection_method == 'noisymin':
                    client_membership = self.noisymin_membership(subset_data, centers[idx], eps)
                elif self.intersection_method == 'ldp':
                    client_membership = self.ldp_membership(subset_data, centers[idx], eps)
                else:
                    raise ValueError
                memberships.append(client_membership)

            # intersection of memberships to get weight
            intersections = self.intersection(memberships)
            intersection_counts = np.array([len(s) for s in intersections])

            # adjust count to be unbiased
            if self.intersection_method == 'ldp':
                intersection_counts = self.ldp_intersection_count_adjust(intersection_counts, eps, parties, local_k)

        elif self.intersection_method == 'fmsketch':
            priv_config = {'eps': eps, 'delta': 1 / n}
            splits = []
            for idx, membership in enumerate(clean_memberships):
                tmp = [np.array(list(m)) for m in membership]
                splits.append(tmp)
            logging.info(f"compute intersection CA with FM sketch...")
            intersection_counts = intersection_ca(n=n,
                                                  splits=splits,
                                                  m=self.config['m'],
                                                  gamma=1.0,
                                                  priv_config=priv_config,
                                                  multithreads=20,
                                                  )
            intersection_counts[intersection_counts < 0] = 0
        elif self.intersection_method == 'allone' or self.intersection_method == 'random':
            intersection_counts = np.ones(shape=np.product([len(c) for c in centers]))
        elif self.intersection_method == 'uniform':
            portions = []
            for m in clean_memberships:
                portions.append([len(members) / n for members in m])
            portions_combines = list(itertools.product(*portions))
            intersection_counts = [n * np.product(p) for p in portions_combines]
        elif self.intersection_method == 'ind_lap':
            portions = []
            for m in clean_memberships:
                portions.append([len(members) / n + np.random.laplace(0, 1/eps) for members in m])
            portions_combines = list(itertools.product(*portions))
            intersection_counts = [n * np.product(p) for p in portions_combines]
        elif self.intersection_method == 'nonpriv':
            clean_intersections = self.intersection(clean_memberships)
            clean_intersection_counts = np.array([len(s) for s in clean_intersections])
            intersection_counts = copy.deepcopy(clean_intersection_counts)
        else:
            raise NotImplementedError

        if run_clean:
            clean_intersections = self.intersection(clean_memberships)
            clean_intersection_counts = np.array([len(s) for s in clean_intersections])
            logging.info(f"(clean) intersection sizes: {clean_intersection_counts}")
            self.clean_intersections = clean_intersection_counts
            return grids, intersection_counts, clean_intersection_counts
        else:
            return grids, intersection_counts, []

    def ldp_membership(self, data, centers, eps):
        local_k = len(centers)
        distance = np.zeros(shape=(data.shape[0], local_k))
        for i, c in enumerate(centers):
            distance[:, i] = np.linalg.norm(data - c, 2, axis=1)
        labels = list(np.argmin(distance, axis=1))

        if local_k > 3 * int(round(np.exp(eps))) + 2:
            # run OLH to perturb labels
            logging.info("===> using OLH for membership")
            perturbed = volh_perturb(labels, eps)
            # decode perturbation and get membership
            memberships = volh_membership(perturbed, domain=local_k, g=int(round(np.exp(eps))) + 1)
        else:
            # run RR to perturb labels
            logging.info("===> using RR for membership")
            perturbed = rr_perturb(labels, eps, local_k)
            # generate rr membership
            memberships = rr_membership(perturbed, local_k)
            # todo: debug
            print(f"*** rr true label histogram: {np.histogram(labels, bins=local_k, range=(0, local_k))}")
            print(f"*** rr perturbed label histogram: {np.histogram(perturbed, bins=local_k, range=(0, local_k))}")

        return memberships

    def ldp_intersection_count_adjust(self, intersection_counts: list, eps: float, parties: int, local_k: int):
        adjusted = np.array(intersection_counts)
        all_combines = [list(range(local_k)) for _ in range(parties)]
        all_combines = list(itertools.product(*all_combines))
        total_bins = np.power(local_k, parties)
        if local_k > 3 * int(round(np.exp(eps))) + 2:
            g = int(round(np.exp(eps))) + 1
            p = np.exp(eps) / (np.exp(eps) + g - 1)
            q = 1.0 / (np.exp(eps) + g - 1)
        else:
            p = np.exp(eps) / (np.exp(eps) + local_k - 1)
            q = 1.0 / (np.exp(eps) + local_k - 1)

        # generate forward probability matrix
        forward_probs = np.ones(shape=(total_bins, total_bins)) * np.power(q, parties)
        for combine in all_combines:
            idx1 = self.cartesian_to_index(combine, local_k)
            for idx2 in range(idx1, np.power(local_k, parties)):
                inner_combine = self.index_to_cartesian(idx2, local_k, parties)
                diff = np.count_nonzero(np.array(combine) != np.array(inner_combine))
                forward_probs[idx1, idx2] = np.power(q, diff) * np.power(p, parties - diff)
                forward_probs[idx2, idx1] = np.power(q, diff) * np.power(p, parties - diff)
                # print(combine, idx1, inner_combine, idx2, forward_probs[idx1, idx2], p, q)
        # print(np.sum(forward_probs, axis=1))
        # exit()

        # compute unbiased frequencies
        inv_prob = np.linalg.inv(forward_probs)
        # todo: debug
        print(f"******* sizes: {inv_prob.shape}, {adjusted.shape}")
        adjusted = np.matmul(inv_prob, adjusted)

        logging.info(f"sum of adjust {np.sum(adjusted)}")
        adjusted[adjusted < 0] = 0

        return adjusted

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

    def save_results(self, losses):
        results = {
            "config": self.config,
            "losses": losses,
            "final_centers": self.centers,
            "private_intersections": self.private_intersections,
            "clean_intersections": self.clean_intersections,
            "clean_centers": self.clean_centers,
        }
        save_result_to_json(results, self.tag, experiment=self.config['dataset'] + '-' \
                                                          + self.config['intersection_method'] + "-VPC")
