import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from utils.test_utils import TestUtils


M = 3
r = 0.3
alpha_list = [0.01, 0.1, 0.3, 0.5, 1, 1.5, 1.8, 2]
num_coh = 1
N_cutoff = 2
N_list = [0,1,2,3,4]
date = '22-02-2022'

fig_iter = 0

for alpha in alpha_list:

    file_name_header = r'../Results/{}_mode/r={}_alpha={}_num_coh={}'.format(M, r, alpha, num_coh)

    for N in N_list:
        n_photon_list = TestUtils.all_possible_n_fix_sum(N_cutoff, N, M)

        file_name_body = r'/N={}'.format(N)
        plt.figure(fig_iter)
        for n_photon in n_photon_list:
            n_photon_str = ''.join(str(e) for e in n_photon)
            for file in glob.glob(
                file_name_header + file_name_body
                + r'/n={}_k=0-{}_{}*.csv'.format(n_photon_str, N, date)
            ):
                results_df = pd.read_csv(file)

            plt.plot(results_df['k'], results_df['prob_error']/results_df['prob_exact'], label='{}'.format(n_photon))

        plt.xlabel('k')
        plt.ylabel('(exact_prob - k_prob)/exact_prob')
        plt.title('M={},r={},alpha={},num_coh={}'.format(M,r,alpha,num_coh,N))
        plt.legend()

        Plot_name = r'../Plots/{}_mode/r={}_alpha={}/Rel_prob_error_num_coh={}_N={}_{}.pdf'\
            .format(M, r, alpha, num_coh, N, date)
        os.makedirs(os.path.dirname(Plot_name), exist_ok=True)

        plt.savefig(Plot_name)
        fig_iter += 1






