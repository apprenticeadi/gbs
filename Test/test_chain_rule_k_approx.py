import math
import pandas as pd
import numpy as np
from scipy.stats import unitary_group
import datetime
import logging
import os
import time

from _walrus_functions import complex_to_real_displacements, reduction, Amat, _prefactor
from chain_rule import *
from utils.log_utils import LogUtils
from utils.run_utils import CircuitUtils
from loop_hafnian_k_approx import loop_hafnian_approx, probability_approx
from loop_hafnian import loop_hafnian

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
    time_array = np.zeros(len(det_outcomes))
    det_pattern_i = det_pattern

    for det_outcome_i in det_outcomes:
        det_pattern_i[mode] = det_outcome_i

        N = np.sum(det_pattern_i)  # total number of photons in output

        start_time = time.time()
        lhaf = loop_hafnian_approx(A=B[:m, :m], gamma=gamma[:m], n=det_pattern_i[:mode + 1], approx=2 * N)
        end_time = time.time()
        probs[det_outcome_i] = (lhaf * np.conj(lhaf)).real / factorial(det_outcome_i)
        time_array[det_outcome_i] = end_time - start_time

    norm_probs = probs.sum()
    probs /= norm_probs

    return probs, time_array


# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
LogUtils.log_config()
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")
date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")

logging.info('Testing how to add k-th order approx to chain rule method, set k=2N, date: {}'.format(date_stamp))

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
M = 8
r = 1.55
alpha = 2
num_coh = 4
cutoff = 10

logging.info('M={}, r={}, alpha={}, num_coh={}, cutoff={}'.format(M, r, alpha, num_coh, cutoff))

U = unitary_group.rvs(M)  # generates random unitary group with dimension M
logging.info('U={}'.format(U))

mu, cov = CircuitUtils.hybrid_gaussian_circuit(M, num_coh, r, alpha, U)
logging.info('mu={}, cov={}'.format(mu, cov))

file_name = r'..\Results\test_chain_k_approx_timed\M={}_num_coh={}_r={}_alpha={}_{}.csv'.format(M, num_coh, r, alpha,
                                                                                          date_stamp)
os.makedirs(os.path.dirname(file_name), exist_ok=True)
results_df = pd.DataFrame(columns=['det_pattern', 'Jake_prob', 'Jake_time', 'k_prob', 'k_time' ,'equal'])

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

overall_Jake_prob = 1

iteration = 0
for mode in range(M):
    m = mode + 1
    gamma -= heterodyne_alpha[mode] * B[:, mode]

    Jake_start_time = time.time()
    probs_Jake = get_prob_Jake(B, gamma, m, det_pattern, cutoff, det_outcomes)
    Jake_end_time = time.time()
    Jake_time = Jake_end_time - Jake_start_time

    logging.info('For {} mode, it takes loop_hafnian_batch {} to produce probability array for all possible outcomes'
                 .format(mode, Jake_time))

    k_start_time = time.time()
    probs_k_approx, k_time_array = get_prob_k_approx(B, gamma, m, det_pattern, cutoff, det_outcomes)
    k_end_time = time.time()
    k_time = k_end_time - k_start_time

    logging.info('For {} mode, it takes loop_hafnian_approx {} to produce probability array for all possible outcomes'
                 .format(mode, k_time))

    for det_outcome_i in det_outcomes:
        det_pattern_i = det_pattern
        det_pattern_i[mode] = det_outcome_i

        Jake_prob = probs_Jake[det_outcome_i]
        k_prob = probs_k_approx[det_outcome_i]
        k_time_i = k_time_array[det_outcome_i]

        equal_bool = np.isclose(Jake_prob, k_prob)

        results_df.loc[iteration] = {
            'det_pattern': '{}'.format(det_pattern_i),
            'Jake_prob': Jake_prob,
            'Jake_time': Jake_time / cutoff,
            'k_prob': k_prob,
            'k_time': k_time_i,
            'equal': equal_bool,
        }
        results_df.to_csv(file_name)

        message = 'det_pattern={}, Jake_prob={}, Jake_time averaged over {} outcomes is {},' \
                  ' k_prob={}, k_time ={}, equal = {}'\
            .format(det_pattern_i, Jake_prob, cutoff, Jake_time/cutoff, k_prob, k_time_i, equal_bool)
        logging.info(message)

        iteration += 1

    det_outcome_choice = np.random.choice(det_outcomes, p=probs_Jake)
    det_pattern[mode] = det_outcome_choice
    overall_Jake_prob = overall_Jake_prob * probs_Jake[det_outcome_choice]

logging.info('Final det pattern = {}, with overall Jake prob = {}'.format(det_pattern, overall_Jake_prob))

total_photons = np.sum(det_pattern)
overall_k_prob = probability_approx(mu=mu, cov=cov, n=det_pattern, approx=2*total_photons)

logging.info('k-th order approx gives prob = {} for k=2N'.format(overall_k_prob))

logging.info('Are they equal= {}'.format(math.isclose(overall_k_prob, overall_Jake_prob)))

logging.info('Calculate loop hafnian for B, gamma, and final det_pattern = {}'.format(det_pattern))

lhafs = loop_hafnian_batch(A=B, D=gamma, fixed_reps=det_pattern[:-1], N_cutoff=cutoff)
logging.info('Jakes chain rule gives a batch of {}'.format(lhafs))

lhaf_exact = loop_hafnian(A=B, D=gamma, reps=det_pattern)
logging.info('A simple loop_hafnian gives {}'.format(lhaf_exact))

lhaf_approx = loop_hafnian_approx(A=B, gamma=gamma, n=det_pattern, approx = 2*total_photons)
logging.info('loop_hafnian_approx gives {} for k=2N'.format(lhaf_approx))

logging.info('lhaf_approx = lhaf_exact is {}'.format(np.isclose(lhaf_approx, lhaf_exact)))

logging.info('lhaf_approx = corresponding entry from lhafs is {}'.format(np.isclose(lhaf_approx, lhafs[det_pattern[-1]])))