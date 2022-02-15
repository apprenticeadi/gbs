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
from loop_hafnian_k_approx import loop_hafnian_approx
from loop_hafnian import loop_hafnian

date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
M = 3  # M must be integer, not numpy.int32, otherwise will cause problems with strawberry fields
r = 1.55  # squeezing magnitude
p = 0  # squeezing angle
alpha = 2.25  # coherent state
phi = 0  # coherent state phase
k_start = 4
k_end = 4
hbar = 2

n_photon = [1, 1, 2]
reps = n_photon + n_photon
N = sum(n_photon)

message = 'Trying to find the error between k-approx and exact hafnian calculation. ' \
          'Run for {} modes, test output pattern is {}, total output photon numbers is {}, ' \
          'k is {} to {}, hbar = {}'.format(M, n_photon, N, k_start, k_end, hbar)
logging.info(message)

message = 'Squeezing r = {}, coherent state alpha = {}'.format(r, alpha)
logging.info(message)

file_name_header = r'../Results/k_approx_error/M={}_N={}'.format(M, N)

# <<<<<<<<<<<< Generate unitary group >>>>>>>>>>>>>>>>>
U = unitary_group.rvs(M)
message = 'U = {}'.format(U)
logging.info(message)

for num_coh in [1]:  # range(2, M):

    results_df = pd.DataFrame(columns=['k', 'lhaf_exact', 'prob_exact', 'lhaf_k_approx', 'prob_k_approx',
                                       'prob_error', 'exact_time', 'k_time'])
    file_name_body = r'/num_coh={}_k={}-{}_{}.csv'.format(num_coh, k_start, k_end,
                                                          date_stamp)
    file_name = file_name_header + file_name_body
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    message = 'Number of coherent states = {}'.format(num_coh)
    logging.info('')
    logging.info(message)

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

    exact_start_time = time.time()
    lhaf_exact = loop_hafnian(A=A, D=gamma, reps=reps)
    exact_end_time = time.time()
    exact_time = exact_end_time - exact_start_time

    prob_exact = lhaf_exact * prob_prefactor
    prob_exact = prob_exact.real

    logging.info('lhaf_exact = {}, prob_exact = {}, exact_time = {}'.format(lhaf_exact, prob_exact, exact_time))

    # lhaf_kisN = loop_hafnian_approx(A=A, gamma=gamma, n=n_photon, approx=2 * N)
    # prob_kisN = lhaf_kisN * prob_prefactor
    # prob_kisN = prob_kisN.real
    #
    # logging.info('lhaf_k for k=N is {}, prob_k = {}'.format(lhaf_kisN, prob_kisN))

    # assert np.isclose(lhaf_exact, lhaf_kisN)

    iteration = 0
    for k in range(k_start, k_end+1):

        start_time = time.time()
        lhaf_k_approx = loop_hafnian_approx(A=A, gamma=gamma, n=n_photon, approx=2 * k)
        end_time = time.time()

        prob_k_approx = lhaf_k_approx * prob_prefactor
        prob_k_approx = prob_k_approx.real

        logging.info('For k = {}, lhaf_k_approx={}, prob_k_approx = {}, time = {}'.format(k, lhaf_k_approx,
                                                                                          prob_k_approx,
                                                                                          end_time - start_time))

        results_df.loc[iteration] = {
            'k': k,
            'lhaf_exact': lhaf_exact,
            'prob_exact': prob_exact,
            'lhaf_k_approx': lhaf_k_approx,
            'prob_k_approx': prob_k_approx,
            'prob_error': prob_exact - prob_k_approx,
            'exact_time': exact_time,
            'k_time': end_time - start_time,
        }

        results_df.to_csv(file_name)

        iteration += 1
