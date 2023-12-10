import numpy as np
# import copy
import itertools
import logging

# from util.save_results import save_result_to_json
# from util.eval_centers import eval_centers
# from util.volh import volh_perturb, volh_membership, rr_membership, rr_perturb
# from util.load_config import generate_local_config
# from util.fmsketch import intersection_ca
# from util.postprocess import norm_sub


class VBase:
    def __init__(self, config, tag, **kwargs):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.config = config
        self.tag = tag
        self.k = config['k']
        self.intersection_method = None


    def noisymin_membership(self, data, centers, eps):
        distance = np.zeros(shape=(data.shape[0], len(centers)))
        for i, c in enumerate(centers):
            distance[:, i] = np.linalg.norm(data - c, 2, axis=1)

        # add noise to distance
        noisy_dist = distance + np.random.laplace(0, 2 / eps, size=distance.shape)

        # select the center with minimum distance for each user data
        noisy_centers = np.argmin(noisy_dist, axis=1)

        client_membership = []
        for i in range(len(centers)):
            client_membership.append(set(np.where(noisy_centers == i)[0]))

        return client_membership

    def clean_membership(self, data, centers, return_assignment=False):
        k = len(centers)
        scores = np.zeros(shape=(data.shape[0], k))
        for i in range(k):
            scores[:, i] = np.linalg.norm(data - centers[i], axis=1)
        labels = np.argmin(scores, axis=1)
        clean_membership = []
        for i in range(len(centers)):
            clean_membership.append(set(np.where(labels == i)[0]))
        if return_assignment:
            return clean_membership, labels
        return clean_membership

    def intersection(self, memberships):
        cartesian = list(itertools.product(*memberships))
        intersections = []
        for combine in cartesian:
            combine = list(combine)
            intersect = combine[0].intersection(*combine[1:])
            intersections.append(intersect)
        return intersections

