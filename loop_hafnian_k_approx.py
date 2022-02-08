import numpy as np
# import numba
from loop_hafnian import _calc_loop_hafnian
from _walrus_functions import complex_to_real_displacements, reduction, Amat
import itertools


def calc_loop_hafnian_approx(A_n, D_n, approx=2, glynn=False):
    """
    Finds an approximate form of the loop hafnian

    Parameters
    ----------
    A_n : matrix that goes into the loop hafnian
    D_n : vector of loop weights (conventionally these go on diagonal of A, I prefer to keep separate)
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
    A_n = np.asarray(A_n, dtype=np.complex128)
    D_n = np.asarray(D_n, dtype=np.complex128)

    N = len(D_n)  # Number of photons  # In fact twice the number of photons

    if approx % 2 == 1:  # if the appoximation order is odd, it does not improve on the even order below it
        approx -= 1  # so might as well reduce it
    if approx == 0:
        return np.prod(D_n)  # 0th order is just the product of D
    if approx >= N:
        # appox order >N is meaningless
        return _calc_loop_hafnian(A_n, D_n, np.ones(N // 2, dtype=np.int64), glynn=glynn)  # loop_hafnian(A,D,glynn=glynn)
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
            Dnew[loops[-1] + 1:] = D_n[loops[-1] + 1:]

            # take submatrices - only keep indices which aren't fixed into loops
            Ds = Dnew[reps == 1]
            As = A_n[reps == 1, :]
            As = As[:, reps == 1]

            # add the product of D for the indices fixed in loops
            # times the loop hafnian of those that aren't
            # loop hafnian function could be replaced with something from thewalrus
            H += np.prod(D_n[loops]) * _calc_loop_hafnian(As, Ds, np.ones(approx // 2, dtype=np.int64), glynn=glynn)
        return H

def loop_hafnian_approx(A, gamma, n, approx=2):
    n2 = n + n
    A_n = reduction(A, n2)  # A reduced according to photon numbers n  # i.e. repeat rows/col n_i times to make A_n
    gamma_n = reduction(gamma, n2)  # gamma reduced according to photon numbers n

    assert len(gamma_n) == np.sum(n2)

    return calc_loop_hafnian_approx(A_n, gamma_n, approx=approx)

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
    prob = loop_hafnian_approx(A, gamma, n, approx)  # find the hafnian (approximately)
    # prob*=_prefactor(mu,cov,hbar) #multiply by prefactpor
    # prob/= np.prod(scipy.special.factorial(n)) #divide by factoials of n
    return prob.real
