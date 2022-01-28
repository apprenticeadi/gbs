import logging
import time
import datetime
import numpy as np
import pandas as pd
from scipy.stats import unitary_group

import strawberryfields as sf
import strawberryfields.ops as ops

from utils.log_utils import LogUtils
from utils.run_utils import CircuitUtils, BenchmarkUtils


# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H:%M:%S.%f)")
LogUtils.log_config()

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
n_samples = 4096
M_list = list(range(2, 5))  # M must be integer, not numpy.int32, otherwise will cause problems with strawberry fields
r = 1.55 # squeezing magnitude
p = 0 # squeezing angle
alpha = 0.6  # coherent state
phi = 0 # coherent state phase
file_name_header = r'..\Results\varying_coh_{}_samples'.format(n_samples)

message = 'Running chain rule sampling with PNRDs for {} modes. n_samples = {}'.format(M_list, n_samples)
logging.info(message)

message = 'Squeezing r = {}, coherent state alpha = {}'.format(r, alpha)
logging.info(message)

# <<<<<<<<<<<< Importing chain rule  >>>>>>>>>>>>>>>>>
message = 'Importing chain rule'
logging.info(message)
from chain_rule import get_samples, get_samples_click
logging.info('')


# <<<<<<<<<<<< Test run with two squeezed modes  >>>>>>>>>>>>>>>>>
test_n_samples = 20
U = unitary_group.rvs(2)  # generates random unitary group with dimension M

logging.info('Test run with two squeezed modes with {} samples'.format(test_n_samples))
logging.info('U={}'.format(U))

mu_test, cov_test = CircuitUtils.hybrid_gaussian_circuit(M=2, num_coh=0, r=r, alpha=alpha, U=U, p=p, phi=phi)
logging.info('mu_test = {}, cov_test = {}'.format(mu, cov))

test_file_name = file_name_header + r'\test_M=2_coh=0_r={}_alpha={}_{}.csv'.format(r, alpha, date_stamp)
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
    coh_ind_ls = [0, 1, int(M/2+0.5), M - 1, M]
    for coh_ind in coh_ind_ls:

        message = 'Number of coherent states = {}'.format(coh_ind)
        logging.info('')
        logging.info(message)

        mu, cov = CircuitUtils.hybrid_gaussian_circuit(M=M, num_coh=coh_ind, r=r, alpha=alpha, U=U, p=p, phi=phi)

        message = 'mu = {}'.format(mu)
        logging.info(message)
        message = 'cov = {}'.format(cov)
        logging.info(message)

        file_name = r'..\Results\varying_coh\M={}_coh={}_r={}_alpha={}_{}.csv'.format(M, coh_ind, r, alpha, date_stamp)
        results_df = pd.DataFrame(columns=['iteration', 'sample', 'time'])

        # chain rule sampling with PNRDs
        iteration = 0
        start_time = time.time()
        for sample in get_samples(mu, cov, n_samples=n_samples):  # This is a generator
            end_time = time.time()
            results_df.loc[iteration] = {
                'iteration': iteration,
                'sample': sample,
                'time': end_time - start_time
            }
            results_df.to_csv(file_name)
            message = 'iteration = {}, sample = {}, time = {}'.format(iteration, sample, end_time - start_time)
            logging.info(message)

            start_time = end_time
            iteration += 1


