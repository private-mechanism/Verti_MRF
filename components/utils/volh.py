# import argparse
import math
import numpy as np
import xxhash
from typing import List, Set


# domain = 0
# epsilon = 0.0
# n = 0
# g = 0
#
# X = []
# Y = []
#
# REAL_DIST = []
# ESTIMATE_DIST = []
#
# p = 0.0
# q = 0.0
#
#
# def generate():
#     # uniform distribution. one can also use other distributions
#     x = np.random.randint(domain)
#     return x
#
#
# def generate_dist():
#     global X, REAL_DIST
#     X = np.zeros(args.n_user, dtype=int)
#     for i in range(args.n_user):
#         X[i] = generate()
#         REAL_DIST[X[i]] += 1
#
#
# def generate_auxiliary():
#     global ESTIMATE_DIST, REAL_DIST, n, p, q, epsilon, domain
#     domain = args.domain
#     epsilon = args.epsilon
#
#     n = args.n_user
#
#     REAL_DIST = np.zeros(domain)
#     ESTIMATE_DIST = np.zeros(domain)
#
#     p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
#     q = 1.0 / (math.exp(epsilon) + g - 1)
#
#
# def perturb():
#     global Y
#     Y = np.zeros(n)
#     for i in range(n):
#         v = X[i]
#         x = (xxhash.xxh32(str(v), seed=i).intdigest() % g)
#         y = x
#
#         p_sample = np.random.random_sample()
#         # the following two are equivalent
#         # if p_sample > p:
#         #     while not y == x:
#         #         y = np.random.randint(0, g)
#         if p_sample > p - q:
#             # perturb
#             y = np.random.randint(0, g)
#         Y[i] = y
#
#     print("perturb done...")
#
#
# def aggregate():
#     global ESTIMATE_DIST
#     ESTIMATE_DIST = np.zeros(domain)
#     for i in range(n):
#         for v in range(domain):
#             if Y[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
#                 ESTIMATE_DIST[v] += 1
#     a = 1.0 * g / (p * g - 1)
#     b = 1.0 * n / (p * g - 1)
#     ESTIMATE_DIST = a * ESTIMATE_DIST - b
#
#
# def error_metric():
#     abs_error = 0.0
#     for x in range(domain):
#         # print REAL_DIST[x], ESTIMATE_DIST[x]
#         abs_error += np.abs(REAL_DIST[x] - ESTIMATE_DIST[x]) ** 2
#     return abs_error / domain
#
#
# def main():
#     generate_auxiliary()
#     generate_dist()
#     results = np.zeros(args.exp_round)
#     for i in range(args.exp_round):
#         perturb()
#         aggregate()
#         results[i] = error_metric()
#     print(np.mean(results), np.std(results))
#
#
# def dispatcher():
#     global g
#     for e in np.arange(2.0):
#
#         print(e, end=' ')
#         args.epsilon = float(e)
#         # try other g
#         g = args.projection_range
#         # OLH
#         g = int(round(math.exp(args.epsilon))) + 1
#         print(g, end=' ')
#         main()

def rr_perturb(labels: List[int], eps: float, domain: int) -> List[int]:
    n = len(labels)
    p = math.exp(eps) / (math.exp(eps) + domain - 1)
    q = 1.0 / (math.exp(eps) + domain - 1)
    p_sample = np.random.random_sample(size=n)
    Y = -1 * np.ones(shape=n)
    for i in range(n):
        if p_sample[i] > p - q:
            # perturb
            Y[i] = np.random.randint(0, domain)
        else:
            Y[i] = labels[i]
    return Y


def rr_membership(perturbed: List[int], domain: int) -> List[Set[int]]:
    n = len(perturbed)
    memberships = []
    for i in range(domain):
        memberships.append(set(np.where(perturbed == i)[0]))
    return memberships


def volh_perturb(labels: List[int], eps: float, domain: int = 0) -> List[int]:
    n = len(labels)
    g = int(round(math.exp(eps))) + 1
    p = math.exp(eps) / (math.exp(eps) + g - 1)
    q = 1.0 / (math.exp(eps) + g - 1)
    p_sample = np.random.random_sample(size=n)
    Y = -1 * np.ones(shape=n)
    for i in range(n):
        y = (xxhash.xxh32(str(labels[i]), seed=i).intdigest() % g)
        if p_sample[i] > p - q:
            # perturb
            y = np.random.randint(0, g)
        Y[i] = y
    return Y


def volh_membership(perturbed: List[int], domain: int, g: int) -> List[Set[int]]:
    n = len(perturbed)
    memberships = [set() for _ in range(domain)]
    for i in range(n):
        for v in range(domain):
            if perturbed[i] == (xxhash.xxh32(str(v), seed=i).intdigest() % g):
                memberships[v].add(i)
    return memberships

# parser = argparse.ArgumentParser(description='Comparisor of different schemes.')
# parser.add_argument('--domain', type=int, default=1024,
#                     help='specify the domain of the representation of domain')
# parser.add_argument('--n_user', type=int, default=10000,
#                     help='specify the number of data point, default 10000')
# parser.add_argument('--exp_round', type=int, default=1,
#                     help='specify the n_userations for the experiments, default 10')
# parser.add_argument('--epsilon', type=float, default=2,
#                     help='specify the differential privacy parameter, epsilon')
# parser.add_argument('--projection_range', type=int, default=2,
#                     help='specify the domain for projection')
# args = parser.parse_args()
# dispatcher()
