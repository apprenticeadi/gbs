import numpy as np
import scipy.special
# import numba
from loop_hafnian import _calc_loop_hafnian
from _walrus_functions import complex_to_real_displacements, reduction, Amat, _prefactor
import itertools


# from strawberryfields_random import random_covariance

def loop_hafnian_approx(A, D, approx=2, glynn=False):
    """
    Finds an approximate form of the loop hafnian

    Parameters
    ----------
    A : matrix that goes into the loop hafnian
    D : vector of loop weights (conventionally these go on diagonal of A, I prefer to keep separate)
    approx : order of approximation
            only keep terms which are at least this order in D,
            i.e. at least this many indices are involved in loops rather than pairs
            The default is 2.
            Odd orders don't improve on the even order below them
    glynn : Use the glynn formula or finite-difference-sieve to calculate loop hafnians
            This improves numerical accuracy with a cost in speed - not that relevant at small scales
            The default is False.

    Returns
    -------
    H, an approximate value of the loop hafnian

    """

    # fix A and D to be 128 bit complex numbers
    A = np.asarray(A, dtype=np.complex128)
    D = np.asarray(D, dtype=np.complex128)

    N = len(D)  # Number of photons

    if approx % 2 == 1:  # if the appoximation order is odd, it does not improve on the even order below it
        approx -= 1  # so might as well reduce it
    if approx == 0:
        return np.prod(D)  # 0th order is just the product of D
    if approx >= N:
        # appox order >N is meaningless
        return _calc_loop_hafnian(A, D, np.ones(N // 2, dtype=np.int64), glynn=glynn)  # loop_hafnian(A,D,glynn=glynn)
    else:
        H = 0
        for output in itertools.combinations(range(N), N - approx):
            # takes all choices of (N-approx) indices to be fixed into loops
            loops = np.asarray(output, dtype=int)

            # make array that is 0 for every index fixed into a loop, 1 for others
            reps = np.ones(N, dtype=int)
            reps[loops] = 0

            # make a temporary version of D
            # only copy the values that come after the last entry in 'loops'
            # this avoids some double counting
            Dnew = np.zeros(N, dtype=np.complex128)
            Dnew[loops[-1] + 1:] = D[loops[-1] + 1:]

            # take submatrices - only keep indices which aren't fixed into loops
            Ds = Dnew[reps == 1]
            As = A[reps == 1, :]
            As = As[:, reps == 1]

            # add the product of D for the indices fixed in loops
            # times the loop hafnian of those that aren't
            # loop hafnian function could be replaced with something from thewalrus
            H += np.prod(D[loops]) * _calc_loop_hafnian(As, Ds, np.ones(approx // 2, dtype=np.int64), glynn=glynn)
        return H


def probability_approx(mu, cov, n, approx=2, hbar=2):
    """
    Approximate probability of a photon number pattern n based on (real) cov matrix and displacment

    Parameters
    ----------
    mu : real displacement vector
    cov : real covariance matrix
    n : photon number pattern (list of integers)
    approx : order of approximation - only keep terms which are at least this order in the displacement
    hbar : value of hbar used in scaling of cov,mu. Default is 2, as in thewalrus

    Returns
    -------
    An approximate probability 'prob'

    """
    A = Amat(cov, hbar)  # find A matrix
    beta = complex_to_real_displacements(mu, hbar=hbar)  # complex displacement vector
    gamma = beta.conj() - A @ beta  # gamma vector
    n2 = n + n
    A_s = reduction(A, n2)  # A reduced according to photon numbers n
    gamma_s = reduction(gamma, n2)  # gamma reduced according to photon numbers n
    prob = loop_hafnian_approx(A_s, gamma_s, approx)  # find the hafnian (approximately)
    # prob*=_prefactor(mu,cov,hbar) #multiply by prefactpor
    # prob/= np.prod(scipy.special.factorial(n)) #divide by factoials of n
    return prob.real

# M=4
# mu=np.random.rand(2*M)
# cov=random_covariance(M)
# A=Amat(cov)
# alpha=complex_to_real_displacements(mu)
# gamma = alpha.conj() - A @ alpha
# haf=loop_hafnian_approx(A, gamma, approx=2*M)
# P=probability_approx(mu,cov,[1]*M,approx=2*M)