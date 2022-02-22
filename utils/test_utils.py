from itertools import product

class TestUtils:

    @staticmethod
    def all_possible_n(N_cutoff, N_sum_max, M):

        n_list = []
        for n_photon in product(range(N_cutoff+1), repeat=M):
            if sum(n_photon) <= N_sum_max:
                n_list.append(list(n_photon))

        return n_list


