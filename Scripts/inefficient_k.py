from math import factorial, comb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
from utils.log_utils import LogUtils

# <<<<<<<<<<<< Logging >>>>>>>>>>>>>>>>>
LogUtils.log_config()
date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

logging.info('Plotting the upper bound for the first k at which k-th order approximation is slower than '
             'directly computing the loop hafnian against N=total number of photons at output')

file_name_head = r'..\Plots'

N_start = 1
N_end = 20

logging.info('N_start = {}, N_end = {}'.format(N_start, N_end))

N_array = np.arange(N_start, N_end + 1, dtype=np.int64)
k_array = np.zeros(len(N_array))

exact_complexities = {}
k_complexities = {}

N_iter = 0
for N in N_array:
    exact_complexity = np.dtype(np.int64)

    exact_complexity = (2 ** N) * ((2 * N) ** 3)
    exact_complexities[N] = np.full(N + 1, exact_complexity)

    logging.info('')
    logging.info('For N = {}, exact complexity is {}'.format(N, exact_complexity))

    k_complexities[N] = np.zeros(N + 1)

    first_k = True
    for k in range(N + 1):
        k_complexity = np.dtype(np.int64)

        k_complexity = comb(2 * N, 2 * k) * (2 ** k) * ((2 * k) ** 3)
        k_complexities[N][k] = k_complexity

        logging.info('For k={}, k complexity is {}'.format(k, k_complexity))

        if k_complexity > exact_complexity and first_k:
            k_array[N_iter] = k
            first_k = False

            logging.info('First inefficient k is {}'.format(k))

    if first_k:
        k_array[N_iter] = k

    N_iter += 1

plt.figure(0)
plt.plot(N_array, k_array)
plt.xlabel('N = total output photon number')
plt.ylabel('k')
plt.xticks(N_array)
plt.title('Upper bound k at which k-approx is slower than exact computation')
plt.savefig(file_name_head + r'\Ineff_k_for_N={}-{}_{}.pdf'.format(N_start, N_end, date_stamp))


for N in N_array:

    plt.figure(N)
    plt.plot(np.arange(N+1), exact_complexities[N], label = 'Exact')
    plt.plot(np.arange(N+1), k_complexities[N], 'x', label= 'k-approx')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('Complexity')
    plt.xticks(np.arange(N+1))
    plt.yscale('log')
    plt.title('Complexity comparison for N = {}'.format(N))
    plt.savefig(file_name_head + r'\Complexity_comparison_for_N={}_{}.pdf'.format(N, date_stamp))

plt.show()

