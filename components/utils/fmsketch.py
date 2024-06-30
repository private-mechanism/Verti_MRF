"""
The code is partially from https://github.com/google-research/privateFM/blob/master/privateFM/FM_simulate.py
"""
import numpy as np
import xxhash
import itertools
import copy
from scipy.stats import hmean
import matplotlib.pyplot as plt
import time
import logging
import concurrent.futures
import pickle


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def gen_fm_sketch(mset, seed, gamma):
    p = gamma / (1 + gamma)
    max_int32 = np.power(2, 32)
    hashed_values = np.array([xxhash.xxh32(user_id.tobytes(), seed=seed).intdigest() for user_id in mset])
    # print(len(mset), np.min(hashed_values / max_int32), np.max(hashed_values / max_int32))
    geometric_values = np.ceil(np.log(hashed_values / max_int32) / np.log(1 - p))
    # print("hist:", np.histogram(geometric_values, bins=20, range=(0, 20)))
    # plt.hist(geometric_values, bins=100)
    # plt.show()
    # exit()
    if len(geometric_values) > 0:
        max_value = np.max(geometric_values)
    else:
        max_value = 0
    return max_value


def set_k_p_min(epsilon, delta, m, gamma):
    """A helper function for computing k_p and eta."""
    if not 0 < epsilon < float('inf') or not 0 < delta < 1:
        k_p = 0
        alpha_min = 0
    else:
        eps1 = epsilon / 4 / np.sqrt(m * np.log(1 / delta))
        k_p = np.ceil(1 / (np.exp(eps1) - 1))
        alpha_min = np.ceil(-np.log(1 - np.exp(-eps1)) / np.log(1 + gamma))
    return k_p, alpha_min


def gen_priv_fm_sketch(mset, seed, gamma, k_p, alpha_min):
    alpha_D = gen_fm_sketch(mset, seed, gamma)
    alpha_p = np.max(np.random.geometric(gamma/(1+gamma), size=int(k_p)))
    return np.maximum(np.maximum(alpha_D, alpha_p), alpha_min)


def complement_fm_sketch(memberships, seed, gamma, priv_config):
    dim = len(memberships)
    fm_sketches = np.zeros(shape=dim)
    for i, membership in enumerate(memberships):
        # generate an fm-sketch for a clustering set
        if priv_config is None:
            max_value = gen_fm_sketch(membership, seed, gamma)
        else:
            k_p, alpha_min = set_k_p_min(priv_config['eps'], priv_config['delta'], priv_config['m'], gamma)
            max_value = gen_priv_fm_sketch(membership, seed, gamma, k_p, alpha_min)
        # update the max for the other sets, so finally a fm-sketch for the complementary sets are generated.
        idxs = list(range(i)) + list(range(i+1, dim))
        fm_sketches[idxs] = np.maximum(max_value, fm_sketches[idxs])
    return max_value, fm_sketches


def one_round_intersection_alpha(splits, seed, gamma, priv_config):
    # compute the sketch of the union of complementary
    sketch = np.zeros(shape=(len(splits), len(splits[0])))
    for i, split in enumerate(splits):
        sketch[i] = complement_fm_sketch(split, seed, gamma, priv_config)
    # take union between different parties
    cartesian = list(itertools.product(*sketch))
    return [np.max(c) for c in cartesian]


def intersection_ca(n, splits, m, gamma, priv_config=None, multithreads=1):
    assert multithreads >= 1
    if (not priv_config is None) :
        assert 'eps' in priv_config and 'delta' in priv_config
        if not 'm' in priv_config:
            priv_config['m'] = m
    seeds = np.random.randint(0, high=100000, size=m)
    num_intersections = np.product([len(s) for s in splits])
    all_sketches = np.zeros(shape=(m, num_intersections))
    start_time = time.time()
    if multithreads == 1:
        for idx, seed in enumerate(seeds):
            if idx % 500 == 0 and idx > 0:
                seconds = time.time() - start_time
                print("iteration:", idx, "time", seconds)
            all_sketches[idx] = one_round_intersection_alpha(splits, seed, gamma, priv_config)
    else:
        logging.info(f"multithreading, # of threads {multithreads}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=multithreads) as executor:
            future_to_seed_idx = {executor.submit(one_round_intersection_alpha,
                                                  splits,
                                                  seed,
                                                  gamma,
                                                  priv_config): idx for idx, seed in enumerate(seeds)}
            for future in concurrent.futures.as_completed(future_to_seed_idx):
                idx = future_to_seed_idx[future]
                try:
                    all_sketches[idx] = future.result()
                except Exception as exc:
                    print(f"Execption when running {idx}-th seed {seeds[idx]}: {exc}")
        seconds = time.time() - start_time
        print("multithreading, run time", seconds)
    debias = 0.7213 / (1 + 1.079 / m)
    if priv_config is None:
        estimate = n - m / np.sum(np.power(1 + gamma, -all_sketches), axis=0) * debias
        print(f"estimate: {estimate}")
    else:
        epsilon, delta = priv_config['eps'], priv_config['delta']
        c = len(splits) * (len(splits[0]) - 1)
        k_p, _ = set_k_p_min(epsilon, delta, m, gamma)
        # the offset (k_p) may need to be revised, because here we are doing the complementary
        raw_comlementary_union = m / np.sum(np.power(1 + gamma, -all_sketches), axis=0) * debias - k_p * c
        # print(k_p, len(splits))
        # print(raw_comlementary_union)
        estimate = n - raw_comlementary_union
        # print(f"DP estimate: {estimate}")
    return estimate


def get_one_set_local_sketches(splits, seed, gamma, priv_config):
    sketches = [np.zeros(shape=len(s)) for s in splits]
    for i, split in enumerate(splits):
        for j, membership in enumerate(split):
            if priv_config is None:
                local_sketch = gen_fm_sketch(membership, seed, gamma)
            else:
                k_p, alpha_min = set_k_p_min(priv_config['eps'], priv_config['delta'], priv_config['m'], gamma)
                local_sketch = gen_priv_fm_sketch(membership, seed, gamma, k_p, alpha_min)
            sketches[i][j] = local_sketch
    return sketches


def get_one_n_two_way_intersection_est(n, splits, m, gamma, priv_config=None, multithreads=1):
    assert multithreads >= 1
    if (not priv_config is None):
        assert 'eps' in priv_config and 'delta' in priv_config
        if not 'm' in priv_config:
            priv_config['m'] = m
    seeds = np.random.randint(0, high=100000, size=m)
    all_one_way_sketches = [np.zeros(shape=(m, len(s))) for s in splits]
    start_time = time.time()
    if multithreads == 1:
        for idx, seed in enumerate(seeds):
            if idx % 500 == 0 and idx > 0:
                seconds = time.time() - start_time
                print("iteration:", idx, "time", seconds)
            new_one_way_sketches = get_one_set_local_sketches(splits, seed, gamma, priv_config)
            for party, sketch in enumerate(new_one_way_sketches):
                all_one_way_sketches[party][idx] = sketch
    else:
        logging.info(f"multithreading, # of threads {multithreads}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=multithreads) as executor:
            future_to_seed_idx = {executor.submit(get_one_set_local_sketches,
                                                  splits,
                                                  seed,
                                                  gamma,
                                                  priv_config): idx for idx, seed in enumerate(seeds)}
            for future in concurrent.futures.as_completed(future_to_seed_idx):
                idx = future_to_seed_idx[future]
                try:
                    new_one_way_sketches = future.result()
                    for party, sketch in enumerate(new_one_way_sketches):
                        all_one_way_sketches[party][idx] = sketch
                except Exception as exc:
                    logging.error(f"Execption when running {idx}-th seed {seeds[idx]}: {exc}")
        seconds = time.time() - start_time
        print("multithreading, run time", seconds)

    # estimate one party's ca
    debias = 0.7213 / (1 + 1.079 / m)
    one_ways = []
    if priv_config:
        epsilon, delta = priv_config['eps'], priv_config['delta']
        k_p, _ = set_k_p_min(epsilon, delta, m, gamma)
    else:
        k_p = 0
    for one_way_sketch in all_one_way_sketches:
        raw_estimate = m / np.sum(np.power(1 + gamma, -one_way_sketch), axis=0) * debias - k_p
        one_ways.append(raw_estimate)

    # estimate two party intersection's ca
    complement_sketches = [np.zeros(shape=(m, len(s))) for s in splits]
    for party, one_way_sketch in enumerate(all_one_way_sketches):
        for i in range(one_way_sketch.shape[1]):
            complement_sketches[party][:, i] = np.max(np.delete(one_way_sketch, i, axis=1), axis=1)
    two_ways = {}
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
            total_intersections = len(splits[i]) * len(splits[j])
            comlementary_union_sketch = np.zeros((m, total_intersections))
            for row in range(m):
                all_sketches = [complement_sketches[i][row], complement_sketches[j][row]]
                combines = np.array(list(itertools.product(*all_sketches)))
                comlementary_union_sketch[row] = np.max(combines, axis=1)
            c = 2 * (len(splits[0]) - 1)
            raw_comlementary_union = m / np.sum(np.power(1 + gamma, -comlementary_union_sketch), axis=0) * debias - k_p * c
            two_ways[(i, j)] = n - raw_comlementary_union
    return one_ways, two_ways


def test_different_ca(repeat=10):
    # intersection_cas = [1000, 2000, 5000, 10000, 15000, 20000, 30000]
    intersection_cas = [1000, 2000]
    result = {}
    for ca in intersection_cas:
        n = ca * 3
        splits_1 = [list(np.arange(start=0, stop=ca * 2)), list(np.arange(start=ca * 2, stop=ca * 3))]
        splits_2 = [list(np.arange(start=0, stop=ca)), list(np.arange(start=ca, stop=ca * 3))]
        splits = [splits_1, splits_2]
        clearn_ca = clean_ca(splits)
        result[ca] = {"clean": clearn_ca, "fm": []}
        priv_config = {'eps': 1, 'delta': 1 / n}
        for _ in range(repeat):
            est = intersection_ca(n=n, splits=splits, m=4096, priv_config=priv_config, gamma=1.0, multithreads=10)
            result[ca]["fm"].append(est)
    f = open('./test_fm.pkl', 'wb')
    pickle.dump(result, f)
    exit()


def clean_ca(splits):
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

if __name__ == '__main__':
    test_different_ca()
    print("testing FM sketch")
    # generate test data
    n = 20000
    k = 4
    ids = np.array(list(range(n)))
    np.random.shuffle(ids)
    splits = []
    splits_points = [10000, 15000, 18000]
    splits.append(copy.deepcopy(np.split(ids, splits_points)))
    np.random.shuffle(ids)
    splits.append(np.split(ids, k))

    cartesian = list(itertools.product(*splits))

    clean_intersections = []
    for combine in cartesian:
        combine = [set(c) for c in combine]
        intersect = combine[0].intersection(*combine[1:])
        clean_intersections.append(intersect)
    clean_ca = [len(s) for s in clean_intersections]
    print("clean intersection size", clean_ca)
    print("sum:", np.sum(clean_ca))
    # exit()
    repeat = 10
    ests = []
    priv_config = {'eps': 1, 'delta': 1e-5}
    for r in range(repeat):
        print("--> repeat", r )
        # m = 1024
        m = 4096
        gamma = 1.0

        estimate = intersection_ca(n, splits, m, gamma,
                                   priv_config=priv_config,
                                   multithreads=10)
        ests.append(estimate)

    print("=== max:", np.max(ests, axis=0), "min:", np.min(ests, axis=0))
    print("=== avg:", np.average(ests, axis=0))
   