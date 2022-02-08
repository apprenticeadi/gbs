####### THIS IS CURRENTLY NOT WORKING!!!


import pandas as pd
import numpy as np
from scipy.stats import unitary_group
import datetime
from loop_hafnian_k_approx import loop_hafnian_approx
import logging
import os

from chain_rule import *
from utils.log_utils import LogUtils
from utils.run_utils import CircuitUtils


def get_prob_Jake(B, gamma, m, det_pattern, cutoff, det_outcomes):
    assert np.array_equal(det_outcomes, np.arange(cutoff + 1))

    mode = m - 1

    lhafs = loop_hafnian_batch(A=B[:m, :m], D=gamma[:m], fixed_reps=det_pattern[:mode], N_cutoff=cutoff)
    probs = (lhafs * lhafs.conj()).real / factorial(det_outcomes)
    norm_probs = probs.sum()
    probs /= norm_probs

    return probs


def get_prob_k_approx(B, gamma, m, det_pattern, cutoff, det_outcomes):
    assert np.array_equal(det_outcomes, np.arange(cutoff + 1))

    mode = m - 1

    probs = np.zeros(len(det_outcomes))
    det_pattern_i = det_pattern

    for det_outcome_i in det_outcomes:
        det_pattern_i[mode] = det_outcome_i

        N = np.sum(det_pattern_i)  # total number of photons in output

        lhaf = loop_hafnian_approx(A=B[:m, :m], gamma=gamma[:m], n=det_pattern_i[:mode + 1], approx=2 * N)

        probs[det_outcome_i] = (lhaf * np.conj(lhaf)).real / factorial(det_outcome_i)

    norm_probs = probs.sum()
    probs /= norm_probs

    return probs


# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
LogUtils.log_config()
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")

logging.info('Testing how to add k-th order approx to chain rule method, date: {}'.format(date_stamp))

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
M = 4
r = 1.55
alpha = 2
num_coh = 0
cutoff = 10

logging.info('M={}, r={}, alpha={}, num_coh={}, cutoff={}'.format(M, r, alpha, num_coh, cutoff))

U = unitary_group.rvs(M)  # generates random unitary group with dimension M
logging.info('U={}'.format(U))

mu, cov = CircuitUtils.hybrid_gaussian_circuit(M, num_coh, r, alpha, U)
logging.info('mu={}, cov={}'.format(mu, cov))

file_name = r'..\Results\test_chain_k_approx\M={}_num_coh={}_r={}_alpha={}_{}.csv'.format(M, num_coh, r, alpha,
                                                                                          time_stamp)
os.makedirs(os.path.dirname(file_name), exist_ok=True)
results_df = pd.DataFrame(columns=['det_pattern', 'Jake_prob', 'k_prob', 'equal'])

assert M == cov.shape[0] // 2

# order = photon_means_order(mu, cov)
# order_inv = invert_permutation(order)
# oo = np.concatenate((order, order + M))
#
# mu = mu[oo]
# cov = cov[np.ix_(oo, oo)]

T, sqrtW = decompose_cov(cov)
chol_T_I = np.linalg.cholesky(T + np.eye(2 * M))
B = Amat(T)[:M, :M]
det_outcomes = np.arange(cutoff + 1)

det_pattern = np.zeros(M, dtype=int)
pure_mu = mu + sqrtW @ np.random.normal(size=2 * M)
pure_alpha = mu_to_alpha(pure_mu)
heterodyne_mu = pure_mu + chol_T_I @ np.random.normal(size=2 * M)
heterodyne_alpha = mu_to_alpha(heterodyne_mu)

gamma = pure_alpha.conj() + B @ (heterodyne_alpha - pure_alpha)

iteration = 0
for mode in range(M):
    m = mode + 1
    gamma -= heterodyne_alpha[mode] * B[:, mode]

    probs_Jake = get_prob_Jake(B, gamma, m, det_pattern, cutoff, det_outcomes)

    probs_k_approx = get_prob_k_approx(B, gamma, m, det_pattern, cutoff, det_outcomes)

    for det_outcome_i in det_outcomes:
        det_pattern_i = det_pattern
        det_pattern_i[mode] = det_outcome_i

        Jake_prob = probs_Jake[det_outcome_i]
        k_prob = probs_k_approx[det_outcome_i]

        results_df.loc[iteration] = {
            'det_pattern': '{}'.format(det_pattern_i),
            'Jake_prob': Jake_prob,
            'k_prob': k_prob,
            'equal': Jake_prob == k_prob,
        }
        results_df.to_csv(file_name)

        message = 'det_pattern={}, Jake_prob={}, k_prob={}, equal = {}'.format(det_pattern_i, Jake_prob, k_prob,
                                                                               Jake_prob == k_prob)
        logging.info(message)

        iteration += 1

    det_outcome_choice = np.random.choice(det_outcomes, p=probs_Jake)
    det_pattern[mode] = det_outcome_choice
