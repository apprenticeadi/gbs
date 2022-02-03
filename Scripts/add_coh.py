import logging
from utils.log_utils import LogUtils

# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
LogUtils.log_config()

import time
import datetime
import numpy as np
import pandas as pd
import os
from scipy.stats import unitary_group

import strawberryfields as sf
import strawberryfields.ops as ops

from utils.run_utils import CircuitUtils, BenchmarkUtils

date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
n_samples = 100
n_repeat = 40
M_list = list(range(8, 10))  # M must be integer, not numpy.int32, otherwise will cause problems with strawberry fields
r = 1.55  # squeezing magnitude
p = 0  # squeezing angle
alpha = 2.25  # coherent state
phi = 0  # coherent state phase

message = 'Running chain rule sampling with PNRDs for {} modes. n_samples = {}'.format(M_list, n_samples)
logging.info(message)

message = 'Squeezing r = {}, coherent state alpha = {}'.format(r, alpha)
logging.info(message)

is_this_test = False
file_name_header = r'..\Results\varying_coh_{}x{}_samples'.format(n_samples, n_repeat)
if is_this_test:
    file_name_header += r'\test'
    logging.info('This is a test run')

# # <<<<<<<<<<<< Importing chain rule  >>>>>>>>>>>>>>>>>
# message = 'Importing chain rule'
# logging.info(message)
# from chain_rule import get_samples, get_samples_click
# logging.info('')


# <<<<<<<<<<<< Test run with 2 squeezed modes  >>>>>>>>>>>>>>>>>
test_n_samples = 20
U = unitary_group.rvs(2)  # generates random unitary group with dimension M

logging.info('Test run with two squeezed modes with {} samples'.format(test_n_samples))
logging.info('U={}'.format(U))

mu_test, cov_test = CircuitUtils.hybrid_gaussian_circuit(M=2, num_coh=0, r=r, alpha=alpha, U=U, p=p, phi=phi)
logging.info('mu_test = {}, cov_test = {}'.format(mu_test, cov_test))

test_file_name = file_name_header + r'\test_M=2_coh=0_r={}_alpha={}_{}.csv'.format(r, alpha, time_stamp)
os.makedirs(os.path.dirname(test_file_name), exist_ok=True)
BenchmarkUtils.chainrule_pnrds(mu=mu_test, cov=cov_test, n_samples=test_n_samples, file_name=test_file_name)

for M in M_list:

    # <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
    message = 'M={} modes'.format(M)
    logging.info('')
    logging.info(message)

    # <<<<<<<<<<<< Generate unitary group >>>>>>>>>>>>>>>>>
    U = unitary_group.rvs(M)  # generates random unitary group with dimension M
    message = 'U = {}'.format(U)
    logging.info(message)

    # <<<<<<<<<<<< Running chain rule for different number of coherent states >>>>>>>>>>>>>>>>>
    coh_ind_ls = [0, 1, int(M / 2 + 0.5), M - 1, M]
    coh_ind_dict = dict.fromkeys(coh_ind_ls)  # removes duplicate

    for coh_ind in coh_ind_dict.keys():
        file_name_body = r'\M={}_coh={}_r={}_alpha={}'.format(M, coh_ind, r, alpha)

        message = 'Number of coherent states = {}'.format(coh_ind)
        logging.info('')
        logging.info(message)

        mu, cov = CircuitUtils.hybrid_gaussian_circuit(M=M, num_coh=coh_ind, r=r, alpha=alpha, U=U, p=p, phi=phi)

        message = 'mu = {}'.format(mu)
        logging.info(message)
        message = 'cov = {}'.format(cov)
        logging.info(message)

        for i_repeat in range(n_repeat):
            message = 'i_repeat = {}'.format(i_repeat)
            logging.info(message)

            file_name = file_name_header + file_name_body + \
                        r'\No_{}_{}.csv'.format(i_repeat, date_stamp)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            BenchmarkUtils.chainrule_pnrds(mu=mu, cov=cov, n_samples=n_samples, file_name=file_name)
