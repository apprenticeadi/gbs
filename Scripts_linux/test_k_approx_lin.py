import logging
from utils.log_utils import LogUtils

# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
LogUtils.log_config_lin()

import time
import datetime
import numpy as np
import pandas as pd
import os
import scipy
from scipy.stats import unitary_group

import strawberryfields as sf
import strawberryfields.ops as ops

from utils.run_utils import CircuitUtils
from _walrus_functions import complex_to_real_displacements, reduction, Amat, _prefactor
from loop_hafnian_k_approx import loop_hafnian_approx, calc_loop_hafnian_approx
from loop_hafnian import loop_hafnian

date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
M = 3  # M must be integer, not numpy.int32, otherwise will cause problems with strawberry fields
r = 1.55  # squeezing magnitude
p = 0  # squeezing angle
alpha = 2.25  # coherent state
phi = 0  # coherent state phase
k_end = 4  # k takes values from 0 to N, when calculating for higher k,  lower k results will be stored as well
hbar = 2
isTest = True

n_photon = [1, 1, 2]
reps = n_photon + n_photon
N = sum(n_photon)

message = 'Trying to find the error between k-approx and exact hafnian calculation. ' \
          'Run for {} modes, test output pattern is {}, total output photon numbers is {}, ' \
          'k is 0 to {}, hbar = {}'.format(M, n_photon, N, k_end, hbar)
logging.info(message)

logging.info('Including the line Dnew[loops[-1] + 1:] = D_n[loops[-1] + 1:]')

message = 'Squeezing r = {}, coherent state alpha = {}'.format(r, alpha)
logging.info(message)

if isTest:
    file_name_header = r'../Results/k_approx_error/test/M={}_N={}_r={}_alpha={}'.format(M, N, r, alpha)
else:
    file_name_header = r'../Results/k_approx_error/M={}_N={}_r={}_alpha={}'.format(M, N, r, alpha)

# <<<<<<<<<<<< Generate unitary group >>>>>>>>>>>>>>>>>
# U = unitary_group.rvs(M)

U = np.asarray(
    [[-0.34860999 + 0.32471167j, -0.07813944 + 0.1676999j, -0.85170116 - 0.11579945j],
     [-0.19282945 + 0.09087308j, -0.92546958 - 0.12856466j, 0.1392731 + 0.24927709j],
     [-0.8323462 - 0.18652639j, 0.2648092 + 0.15041356j, 0.22615505 + 0.35848773j]],
    dtype=np.complex128)

message = 'U = {}'.format(U)
logging.info(message)

for num_coh in [1]:  # range(2, M):

    # <<<<<<<<<<<< Result file >>>>>>>>>>>>>>>>>
    results_df = pd.DataFrame(columns=['k', 'lhaf_exact', 'prob_exact', 'lhaf_k_approx', 'prob_k_approx',
                                       'prob_error', 'exact_time', 'k_time'])
    if isTest:
        file_name_body = r'/num_coh={}_k=0-{}_{}.csv'.format(num_coh, k_end, time_stamp)
    else:
        file_name_body = r'/num_coh={}_k=0-{}_{}.csv'.format(num_coh, k_end, date_stamp)

    file_name = file_name_header + file_name_body
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    message = 'Number of coherent states = {}'.format(num_coh)
    logging.info('')
    logging.info(message)

    # <<<<<<<<<<<< Generate A matrix and gamma vector >>>>>>>>>>>>>>>>>
    mu, cov = CircuitUtils.hybrid_gaussian_circuit(M=M, num_coh=num_coh, r=r, alpha=alpha, U=U, p=p, phi=phi)

    message = 'mu = {}'.format(mu)
    logging.info(message)
    message = 'cov = {}'.format(cov)
    logging.info(message)

    prob_prefactor = _prefactor(mu, cov, hbar) / np.prod(scipy.special.factorial(n_photon))

    A = Amat(cov, hbar)  # find A matrix
    beta = complex_to_real_displacements(mu, hbar=hbar)  # complex displacement vector
    gamma = beta.conj() - A @ beta  # gamma vector

    logging.info('A={}, gamma = {}'.format(A, gamma))

    # <<<<<<<<<<<< Calculate exact lhaf and prob >>>>>>>>>>>>>>>>>
    exact_start_time = time.time()
    lhaf_exact = loop_hafnian(A=A, D=gamma, reps=reps)
    exact_end_time = time.time()
    exact_time = exact_end_time - exact_start_time

    prob_exact = lhaf_exact * prob_prefactor
    prob_exact = prob_exact.real

    logging.info('lhaf_exact = {}, prob_exact = {}, exact_time = {}'.format(lhaf_exact, prob_exact, exact_time))

    # <<<<<<<<<<<< Run loop_hafnian_approx, but save intermediate k results >>>>>>>>>>>>>>>>>
    assert A.shape[0] == 2 * len(n_photon)
    assert len(gamma) == A.shape[0]

    reps = np.concatenate([n_photon, n_photon])

    A_n = reduction(A, reps)
    gamma_n = reduction(gamma, reps)

    approx_time = 0
    lhaf_approx = 0
    for k_iter in range(k_end + 1):
        approx = 2 * k_iter

        start_time = time.time()
        H = calc_loop_hafnian_approx(A_n, gamma_n, approx=approx)
        end_time = time.time()
        approx_time += end_time - start_time

        logging.info('k={} term gives H = {}'.format(k_iter, H))
        logging.info('')

        lhaf_approx += H

        prob_approx = lhaf_approx * prob_prefactor
        prob_approx = prob_approx.real

        results_df.loc[k_iter] = {
            'k': k_iter,
            'lhaf_exact': lhaf_exact,
            'prob_exact': prob_exact,
            'lhaf_k_approx': lhaf_approx,
            'prob_k_approx': prob_approx,
            'prob_error': prob_exact - prob_approx,
            'exact_time': exact_time,
            'k_time': approx_time,
        }
        results_df.to_csv(file_name)
