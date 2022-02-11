import logging
from utils.log_utils import LogUtils

# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
LogUtils.log_config()

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
M = 8  # M must be integer, not numpy.int32, otherwise will cause problems with strawberry fields
r = 1.55  # squeezing magnitude
p = 0  # squeezing angle
alpha = 2.25  # coherent state
phi = 0  # coherent state phase
k_cutoff = 4
hbar = 2

n_photon = [1, 3, 2, 1, 3, 2, 2, 2]
reps = n_photon + n_photon
N = sum(n_photon)

message = 'Trying to find the error between k-approx and exact hafnian calculation. ' \
          'Run for {} modes, test output pattern is {}, total output photon numbers is {}, ' \
          'cutoff k is {}, hbar = {}'.format(M, n_photon, N, k_cutoff, hbar)
logging.info(message)

message = 'Squeezing r = {}, coherent state alpha = {}'.format(r, alpha)
logging.info(message)

file_name_header = r'..\Results\k_approx_error\M={}_N={}'.format(M, N)

# <<<<<<<<<<<< Generate unitary group >>>>>>>>>>>>>>>>>
U = np.array([[0.07710871 + 0.21162963j, 0.27311514 + 0.06344089j, 0.0801365 - 0.09423754j,
      -0.23360802 + 0.34126126j, 0.07158522 + 0.09883681j, -0.32271203 - 0.39770831j,
      0.25101402 + 0.03198496j, 0.32266628 - 0.48883871j],
     [0.32788034 - 0.30682031j, 0.23932279 - 0.15088125j, 0.37687963 + 0.28972029j,
      -0.32862099 - 0.03917891j, -0.2744999 + 0.10868782j, -0.16949767 + 0.21727285j,
      0.3700108 - 0.05702261j, 0.01491367 + 0.28165619j],
     [-0.12704007 + 0.12526426j, 0.05834088 + 0.03415773j, 0.36085529 + 0.47316356j,
      0.24651493 + 0.24854646j, -0.19325134 - 0.37671468j, 0.39266907 - 0.16065757j,
      0.04122026 - 0.29235736j, -0.07614709 - 0.18634334j],
     [-0.09706315 - 0.26259648j, -0.45025059 + 0.1955157j, 0.3779968 + 0.11114862j,
      -0.25434377 - 0.24168248j, 0.05687579 - 0.17080884j, -0.20788559 + 0.14577352j,
      -0.17395895 + 0.20681379j, -0.05173179 - 0.47931867j],
     [-0.13446759 - 0.27447351j, 0.37285055 - 0.10290388j, 0.12081865 - 0.12422378j,
      0.39784465 + 0.09445783j, -0.07175215 - 0.33983902j, -0.08259578 + 0.16232056j,
      -0.08634183 + 0.54244939j, 0.32212762 + 0.02152336j],
     [0.17590852 + 0.02865416j, -0.47492932 - 0.35465557j, 0.12562331 - 0.0285822j,
      -0.00704497 + 0.3860293j, 0.14686335 - 0.05721707j, 0.10183017 - 0.27666554j,
      0.22071533 + 0.45456591j, -0.21016518 + 0.19989904j],
     [-0.13139794 - 0.1365314j, 0.28652845 - 0.06748636j, 0.23589884 - 0.09749739j,
      0.01135504 - 0.38053888j, 0.67130109 - 0.16424714j, 0.06457616 - 0.25113089j,
      0.25583464 - 0.09826983j, -0.20447936 + 0.07462442j],
     [0.4683263 + 0.50694829j, -0.07080523 + 0.05664734j, 0.2821849 - 0.24078259j,
      0.05025394 - 0.10029979j, 0.20208134 - 0.1494401j, 0.21071003 + 0.43077042j,
      0.05713519 - 0.01339182j, 0.26034667 - 0.03010458j]])
message = 'U = {}'.format(U)
logging.info(message)

for num_coh in [2]: # range(2, M):

    results_df = pd.DataFrame(columns=['k', 'lhaf_exact', 'prob_exact', 'lhaf_k_approx', 'prob_k_approx',
                                       'prob_error', 'exact_time', 'k_time'])
    file_name_body = r'\num_coh={}_{}.csv'.format(num_coh, date_stamp)
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

    # exact_start_time = time.time()
    # lhaf_exact = loop_hafnian(A=A, D=gamma, reps=reps)
    # exact_end_time = time.time()
    # exact_time = exact_end_time - exact_start_time
    #
    # prob_exact = lhaf_exact * prob_prefactor
    # prob_exact = prob_exact.real
    #
    # logging.info('lhaf_exact = {}, prob_exact = {}, exact_time = {}'.format(lhaf_exact, prob_exact, exact_time))
    #
    # lhaf_kisN = loop_hafnian_approx(A=A, gamma=gamma, n=n_photon, approx=2 * N)
    # prob_kisN = lhaf_kisN * prob_prefactor
    # prob_kisN = prob_kisN.real
    #
    # logging.info('lhaf_k for k=N is {}, prob_k = {}'.format(lhaf_kisN, prob_kisN))

    # assert np.isclose(lhaf_exact, lhaf_kisN)

    iteration = 0
    for k in [1]:  # range(k_cutoff):

        start_time = time.time()
        lhaf_k_approx = loop_hafnian_approx(A=A, gamma=gamma, n=n_photon, approx=2 * k)
        end_time = time.time()

        prob_k_approx = lhaf_k_approx * prob_prefactor
        prob_k_approx = prob_k_approx.real

        logging.info('For k = {}, lhaf_k_approx={}, prob_k_approx = {}, time = {}'.format(k, lhaf_k_approx,
                                                                                          prob_k_approx,
                                                                                          end_time - start_time))

        # results_df.loc[iteration] = {
        #     'k': k,
        #     'lhaf_exact': lhaf_exact,
        #     'prob_exact': prob_exact,
        #     'lhaf_k_approx': lhaf_k_approx,
        #     'prob_k_approx': prob_k_approx,
        #     'prob_error': prob_exact - prob_k_approx,
        #     'exact_time': exact_time,
        #     'k_time': end_time - start_time,
        # }
        #
        # results_df.to_csv(file_name)
        #
        iteration += 1

