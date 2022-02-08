import numpy as np
from scipy.stats import unitary_group
import strawberryfields as sf
import strawberryfields.ops as ops
from loop_hafnian import loop_hafnian
from loop_hafnian_k_approx import loop_hafnian_approx
from _walrus_functions import *

M = 4
eng = sf.Engine(backend='gaussian')
prog = sf.Program(M) # creates an M mode quantum program
U = unitary_group.rvs(M) # generates random unitary group with dimension M

r = 1.55
eta = 1
hbar = 2

with prog.context as q:
    for i in range(M):
        ops.Sgate(r) | q[i] # Append S gate to each mode. S gate is phase space squeezing gate. r is the squeezing amount.
        # ops.LossChannel(eta) | q[i] # Append a loss channel to each mode. eta is transmissivity
    ops.Interferometer(U) | q # default is rectangular mesh, the Clements design

state = eng.run(prog).state # I think this can be understood as the quantum state after computation. With a gaussian backend we get a gaussian state.

# get wigner function displacement and covariance
mu = state.means() # The vector of means describing the Gaussian state.
cov = state.cov() # The covariance matrix describing the Gaussian state.

A = Amat(cov, hbar)  # find A matrix
beta = complex_to_real_displacements(mu, hbar=hbar)  # complex displacement vector
gamma = beta.conj() - A @ beta  # gamma vector

n = [1] * M
reps = n + n

lhaf_exact = loop_hafnian(A, gamma, reps)
lhaf_approx = loop_hafnian_approx(A, gamma, n, approx = np.sum(reps))


print('lhaf_exact = {}'.format(lhaf_exact))
print('lhaf_approx = {}'.format(lhaf_approx))
print('equal = {}'.format(lhaf_exact == lhaf_approx))