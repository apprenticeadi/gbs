import numpy as np 
from loop_hafnian_batch import loop_hafnian_batch
from loop_hafnian_batch_gamma import loop_hafnian_batch_gamma
from scipy.special import factorial
from strawberryfields.decompositions import williamson
from thewalrus.quantum import (
    Amat,
    Qmat,
    photon_number_mean_vector, 
    mean_clicks,
    reduced_gaussian
    )

import time
import logging
from utils.log_utils import LogUtils

def decompose_cov(cov):
    # Williamson decomposition.

    m = cov.shape[0] // 2
    D, S = williamson(cov)  # Williamson decomposition, any positive definite real matrix V=SDS^T
    T = S @ S.T # This code works with hbar=2
    DmI = D - np.eye(2*m)
    DmI[abs(DmI) < 1e-11] = 0. # remove slightly negative values
    sqrtW = S @ np.sqrt(DmI)
    return T, sqrtW

def mu_to_alpha(mu, hbar=2):
    M = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:M] + 1j * mu[M:]) / np.sqrt(2 * hbar)
    return alpha

def invert_permutation(p):
    s = np.empty_like(p, dtype=int)
    # Returns array of uninitialized (arbitrary) data with the same shape and type as p.

    s[p] = np.arange(p.size, dtype=int)
    # If I have an array s = np.array([5,3,7]), and p = np.array([0,2,1]), then s[p] returns s in order given by p, i.e.
    # s[p] = np.array([5,7,3]).
    # So what this line does is, s in order of p is labelled 0 to p.size.
    # e.g. p = [1,3,4,2,0], then s[1]=0, s[3]=1, s[4]=2, s[2]=3, s[0]=4, so returns s= [4,0,3,1,2]

    return s

def photon_means_order(mu, cov):
    # Returns array of mode number in ascending order of mean photon number

    means = photon_number_mean_vector(mu, cov)  # Calculate the mean photon number of each of the modes in a Gaussian state
    order = [x for _, x in sorted(zip(means, range(len(means))))]
    return np.asarray(order)

def click_means_order(cov):

    M = cov.shape[0] // 2 
    mu = np.zeros(2*M)

    means = np.zeros(M)

    for i in range(M):
        mu_i, cov_i = reduced_gaussian(mu, cov, [i])
        means[i] = mean_clicks(cov_i)

    order = [x for _, x in sorted(zip(means, range(len(means))))]
    return np.asarray(order)

def get_samples(mu, cov, cutoff=10, n_samples=10):
    # Retrieves number of modes. since covariance matrix is 2M by 2M.
    M = cov.shape[0] // 2 # integer division, quotient without remainder

    order = photon_means_order(mu, cov)
    # Returns array of mode number in ascending order of mean photon number
    # Order with which the chain rule algorithm progresses is arbitrary, so choose to go in order of increasing mean
    # photon/click number
    # This slightly reduces run time, since photons less likely to be detected in earlier modes, and size of loop
    # hafnians in these stages generally reduced.

    order_inv = invert_permutation(order)

    oo = np.concatenate((order, order+M))
    # e.g. order = [2,1,3,0], then oo = [2,1,3,0,2+4, 1+4, 3+4, 0+4]

    # rearrange mu and cov in order of oo, i.e. in order of ascending mean photon number
    mu = mu[oo]
    cov = cov[np.ix_(oo, oo)]

    T, sqrtW = decompose_cov(cov)
    # Williamson decomposition of mixed state as convex combination of pure states
    # Continue algorithm with pure state given by covariance matrix T

    chol_T_I = np.linalg.cholesky(T+np.eye(2*M))
    B = Amat(T)[:M,:M]
    # Amat returns the matrix of the Gaussian state whose hafnian gives the photon number
    # probabilities. And because we're dealing with pure states now, A is block diagonal matrix and we just need to compute lhaf of B
    det_outcomes = np.arange(cutoff+1)

    for i in range(n_samples):
        det_pattern = np.zeros(M, dtype=int)
        pure_mu = mu + sqrtW @ np.random.normal(size=2*M)
        # Get the vector of means corresponding to the pure state. (After williamson decomposition we are dealing with
        # pure states)
        pure_alpha = mu_to_alpha(pure_mu) # Finding the mean displacement of each mode from mu.
        heterodyne_mu = pure_mu + chol_T_I @ np.random.normal(size=2*M)
        heterodyne_alpha = mu_to_alpha(heterodyne_mu)
       
        gamma = pure_alpha.conj() + B @ (heterodyne_alpha - pure_alpha)

        message = 'For {}th sample'.format(i)
        logging.info('')
        logging.info(message)

        for mode in range(M):
            m = mode + 1
            gamma -= heterodyne_alpha[mode] * B[:, mode]

            message = 'For {}th mode'.format(mode)
            logging.info(message)

            start_time = time.time()
            lhafs = loop_hafnian_batch(B[:m,:m], gamma[:m], det_pattern[:mode], cutoff)
            # This should be the most costly step
            end_time = time.time()

            message = 'Run time = {} for loop_hafnian_batch'.format(mode, end_time-start_time)
            logging.info(message)
            logging.info('lhafs = {}'.format(lhafs))

            probs = (lhafs * lhafs.conj()).real / factorial(det_outcomes)
            norm_probs = probs.sum()
            probs /= norm_probs 

            det_outcome_i = np.random.choice(det_outcomes, p=probs)
            det_pattern[mode] = det_outcome_i

        yield det_pattern[order_inv]

def get_heterodyne_fanout(alpha, fanout):
    M = len(alpha)

    alpha_fanout = np.zeros((M, fanout), dtype=np.complex128)
    for j in range(M):
        alpha_j = np.zeros(fanout, dtype=np.complex128)
        alpha_j[0] = alpha[j] # put the coherent state in 0th mode 
        alpha_j[1:] = (np.random.normal(size=fanout-1) +
                 1j * np.random.normal(size=fanout-1))

        alpha_fanout[j,:] = np.fft.fft(alpha_j, norm='ortho')

    return alpha_fanout

def get_samples_click(mu, cov, cutoff=1, fanout=10, n_samples=10):

    M = cov.shape[0] // 2

    order = photon_means_order(mu, cov)
    order_inv = invert_permutation(order)
    oo = np.concatenate((order, order+M))

    mu = mu[oo]
    cov = cov[np.ix_(oo, oo)]
    T, sqrtW = decompose_cov(cov)
    chol_T_I = np.linalg.cholesky(T+np.eye(2*M))   
    B = Amat(T)[:M,:M] / fanout

    det_outcomes = np.arange(cutoff+1)

    for i in range(n_samples):
        det_pattern = np.zeros(M, dtype=int)
        click_pattern = np.zeros(M, dtype=np.int8)
        fanout_clicks = np.zeros(M, dtype=int)

        pure_mu = mu + sqrtW @ np.random.normal(size=2*M)
        pure_alpha = mu_to_alpha(pure_mu)
        het_mu = pure_mu + chol_T_I @ np.random.normal(size=2*M)
        het_alpha = mu_to_alpha(het_mu)

        het_alpha_fanout = get_heterodyne_fanout(het_alpha, fanout)
        het_alpha_sum = het_alpha_fanout.sum(axis=1)

        gamma = (pure_alpha.conj() / np.sqrt(fanout) + 
                    B @ (het_alpha_sum - np.sqrt(fanout) * pure_alpha))
        gamma_fanout = np.zeros((fanout, M), dtype=np.complex128)

        for mode in range(M):
            gamma_fanout[0,:] = gamma - het_alpha_fanout[mode, 0] * B[:, mode]
            for k in range(1, fanout):
                gamma_fanout[k,:] = gamma_fanout[k-1,:] - het_alpha_fanout[mode,k] * B[:,mode]
            lhafs = loop_hafnian_batch_gamma(B[:mode+1,:mode+1], gamma_fanout[:,:mode+1], 
                                            det_pattern[:mode], cutoff)
            probs = (lhafs * lhafs.conj()).real / factorial(det_outcomes)

            for k in range(fanout):
                gamma = gamma_fanout[k,:]
                probs_k = probs[k,:] / probs[k,:].sum()
                det_outcome = np.random.choice(det_outcomes, p=probs_k)
                det_pattern[mode] += det_outcome
                if det_outcome > 0:
                    click_pattern[mode] = 1
                    fanout_clicks[mode] = k
                    break 

        yield click_pattern[order_inv]
