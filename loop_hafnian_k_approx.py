import logging
import pandas as pd
import numpy as np
import scipy
from loop_hafnian import _calc_loop_hafnian, loop_hafnian
from _walrus_functions import complex_to_real_displacements, reduction, Amat, _prefactor
import itertools

# TODO: is calc_loop_hafnian_approx calculating just the k-th order or up to k-th order?
# Only k-th order

def calc_loop_hafnian_approx(A_n, D_n, approx=2, glynn=False):
    """
    Finds the k-th order term in the k-th order expansion of the loop hafnian

    Parameters
    ----------
    A_n : matrix that goes into the loop hafnian, shape 2N*2N, N=total photon number
    D_n : vector of loop weights (conventionally these go on diagonal of A, I prefer to keep separate), shape 2N
    approx : 2k, order of approximation, from 0 to 2N
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


    # This line is very problematic. If we feed in A matrix, N is twice the number of photons and everything is fine.
    # But if we feed in the B matrix, then N is the number of photons and everything that follows is wrong...
    N2 = len(D_n)  # Twice the number of photons

    if approx % 2 == 1:  # if the appoximation order is odd, it does not improve on the even order below it
        approx -= 1  # so might as well reduce it
    if approx == 0:
        return np.prod(D_n)  # 0th order is just the product of D
    if approx > N2: # For now change to >N, so can check if approx=N gives correct answer
        # appox order >N is meaningless
        return loop_hafnian(A_n, D_n, glynn=glynn) # _calc_loop_hafnian(A_n, D_n, np.ones(N // 2, dtype=np.int64), glynn=glynn)
    else:
        H = 0
        for output in itertools.combinations(range(N2), N2 - approx):
            # takes all choices of (N-approx) indices to be fixed into loops
            # This gives the indices of gamma product
            # If approx=N2, output = ()
            loops = np.asarray(output, dtype=int)

            # make array that is 0 for every index fixed into a loop, 1 for others
            reps = np.ones(N2, dtype=int)
            reps[loops] = 0

            # make a temporary version of D
            # only copy the values that come after the last entry in 'loops'
            # this avoids some double counting
            # This block doesn't make sense...
            # No D vector should be fed into the *hafnian* calculation, especially when the code is using a
            # lhaf function to do this, D should be set to all zero.
            Dnew = np.zeros(N2, dtype=np.complex128)
            if len(loops) == 0:
                pass
            else:
                Dnew[loops[-1] + 1:] = D_n[loops[-1] + 1:] # this line wouldn't work for approx=N2

            # take submatrices - only keep indices which aren't fixed into loops
            Ds = Dnew[reps == 1]
            As = A_n[reps == 1, :]
            As = As[:, reps == 1]

            # add the product of D for the indices fixed in loops
            # times the loop hafnian of those that aren't
            # loop hafnian function could be replaced with something from thewalrus
            # Ds should be all zero here, lhaf of a diagonally zero matrix is the hafnian of the matrix
            haf_term = loop_hafnian(As, Ds, glynn = glynn)
            gamma_prod = np.prod(D_n[loops])
            H_term = gamma_prod * haf_term

            logging.info('k={},output={},gamma_prod={},haf={}'.format(approx/2, output, gamma_prod, haf_term))

            H += H_term
                 #_calc_loop_hafnian(As, Ds, np.ones(approx // 2, dtype=np.int64), glynn=glynn)
        return H

def loop_hafnian_approx(A, gamma, n, k=1):
    """
    Calculates the loop hafnian *to* the k-th order approximation for a mixed state.
    Cannot accept B matrix for pure state yet
    Args:
        A: A matrix, shape 2M * 2M, where M is number of modes
        gamma: displacement vector, shape 2m
        n: output photon pattern, length M, sum(n)=N
        k: order of approximation, from 0 to N

    Returns:
        lhaf_approx, the k-th order approximation of the loop hafnian
    """

    assert A.shape[0] == 2 * len(n)
    assert len(gamma) == A.shape[0]

    if A.shape[0] == 2 * len(n):
        n = np.concatenate([n,n])

    A_n = reduction(A, n)  # A reduced according to photon numbers n  # i.e. repeat rows/col n_i times to make A_n
    gamma_n = reduction(gamma, n)  # gamma reduced according to photon numbers n

    lhaf_approx = 0
    for k_iter in range(k+1):
        approx = 2*k_iter
        H = calc_loop_hafnian_approx(A_n, gamma_n, approx=approx)
        lhaf_approx += H

    return lhaf_approx


def direct_sum(A, B):
    # A and B are complex matrices!
    dsum = np.zeros(np.add(A.shape, B.shape), dtype=complex)
    dsum[:A.shape[0], :A.shape[1]] = A
    dsum[A.shape[0]:, A.shape[1]:] = B

    return dsum


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
    prob*=_prefactor(mu,cov,hbar) #multiply by prefactpor
    prob/= np.prod(scipy.special.factorial(n)) #divide by factoials of n
    return prob.real
