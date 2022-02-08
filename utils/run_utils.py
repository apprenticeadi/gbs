import numpy as np
import pandas as pd
import time
from scipy.stats import unitary_group
import strawberryfields as sf
import strawberryfields.ops as ops
import logging

from chain_rule import get_samples

class CircuitUtils:

    @staticmethod
    def hybrid_gaussian_circuit(M, num_coh, r, alpha, U, p=0, phi=0):
        """
        :param M: Number of modes
        :param num_coh: Number of modes populated by coherent state
        :param r: Squeezing magnitude
        :param alpha: Coherent state amplitude
        :param U: Unitary for interferometer
        :param p: Squeezing angle
        :param phi: Coherent state phase
        """

        eng = sf.Engine(backend='gaussian')
        # The engine cannot be reused between programs.
        # Otherwise will automatically concatenate the next program onto the previous one.
        prog = sf.Program(M)
        with prog.context as q:
            for i in range(0, num_coh):
                ops.Coherent(r=alpha, phi=phi) | q[i]  # Sets mode into coherent state D(alpha)|0>

            for i in range(num_coh, M):
                ops.Squeezed(r=r, p=p) | q[i]  # Sets mode into single mode squeezed state S(z)|0>

            ops.Interferometer(U) | q

        state = eng.run(prog).state
        # I think this can be understood as the quantum state after computation.
        # With a gaussian backend we get a gaussian state.

        mu = state.means()  # The vector of means describing the Gaussian state.
        cov = state.cov()  # The covariance matrix describing the Gaussian state.

        return mu, cov


class BenchmarkUtils:

    @staticmethod
    def chainrule_pnrds(mu, cov, n_samples, file_name):

        results_df = pd.DataFrame(columns=['iteration', 'sample', 'time'])

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


