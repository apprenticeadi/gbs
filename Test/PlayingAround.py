import numpy as np
from scipy.stats import unitary_group
import strawberryfields as sf
import strawberryfields.ops as ops
from chain_rule import get_samples, get_samples_click
from thewalrus.quantum import photon_number_mean_vector, mean_clicks
from MIS import MIS_IPS
from MIS_click import ClickMIS

M = 4
eng = sf.Engine(backend='gaussian')
prog = sf.Program(M) # creates an M mode quantum program
U = unitary_group.rvs(M) # generates random unitary group with dimension M

r = 1
eta = 0.9
alpha = 0.1
with prog.context as q:
    for i in range(M):
        ops.Sgate(r) | q[i] # Append S gate to each mode. S gate is phase space squeezing gate. r is the squeezing amount.
        ops.LossChannel(eta) | q[i] # Append a loss channel to each mode. eta is transmissivity
    ops.Interferometer(U) | q # default is rectangular mesh, the Clements design

state = eng.run(prog).state # I think this can be understood as the quantum state after computation. With a gaussian backend we get a gaussian state.

# get wigner function displacement and covariance
mu = state.means() # The vector of means describing the Gaussian state.
cov = state.cov() # The covariance matrix describing the Gaussian state.

# chain rule sampling with PNRDs

# for sample in get_samples(mu, cov, n_samples=10):
#     print(sample)

# chain rule sampling with threshold detectors

# for sample in get_samples_click(mu, cov, n_samples=10):
#     print(sample)
#
# MIS with PNRDs
N = int(np.round(photon_number_mean_vector(mu, cov)).sum()) # Calculate the mean photon number of each of the modes in a Gaussian state
mis = MIS_IPS(cov, N)
mis.run_chain(120)

burn_in = 20
thinning_rate = 10

print(np.array(mis.chain_patterns[burn_in::thinning_rate]))

# # MIS with threshold detectors
# N = int(np.round(mean_clicks(cov)))
# mis = ClickMIS(cov, N)
# mis.run_chain(120)
#
# burn_in = 20
# thinning_rate = 10
#
# print(np.array(mis.chain_patterns[burn_in::thinning_rate]))