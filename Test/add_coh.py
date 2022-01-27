import logging
import time
from utils.log_utils import LogUtils
import datetime

n_samples = 100

M_list = list(range(2, 9))  # M must be integer, not numpy.int32, otherwise will cause problems with strawberry fields

# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")
LogUtils.log_config()
message = 'Running chain rule sampling with PNRDs for {} modes. n_samples = {}'.format(M_list, n_samples)
logging.info(message)

import numpy as np
from scipy.stats import unitary_group
import strawberryfields as sf
import strawberryfields.ops as ops
import pandas as pd

# <<<<<<<<<<<< Importing chain rule  >>>>>>>>>>>>>>>>>
message = 'Importing chain rule'
logging.info(message)

from chain_rule import get_samples, get_samples_click

# <<<<<<<<<<<< Parameters >>>>>>>>>>>>>>>>>
r = 1.55 # squeezing magnitude
alpha = 0.6  # coherent state

message = 'Squeezing r = {}, coherent state alpha = {}'.format(r, alpha)
logging.info(message)


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
    coh_ind_ls = [0, 1, int(M/2)+1, M - 1, M]
    for coh_ind in coh_ind_ls:

        message = 'Number of coherent states = {}'.format(coh_ind)
        logging.info('')
        logging.info(message)

        eng = sf.Engine(backend='gaussian')
        prog = sf.Program(M)
        with prog.context as q:
            for i in range(0, coh_ind):
                ops.Coherent(r=alpha) | q[i]

            for i in range(coh_ind, M):
                ops.Squeezed(r=r) | q[i]

            ops.Interferometer(U) | q

        state = eng.run(
            prog).state  # I think this can be understood as the quantum state after computation. With a gaussian backend we get a gaussian state.

        # get wigner function displacement and covariance
        mu = state.means()  # The vector of means describing the Gaussian state.
        cov = state.cov()  # The covariance matrix describing the Gaussian state.

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


