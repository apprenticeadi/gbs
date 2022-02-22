import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

M = 3
N = 4
n_photon=[1,1,2]
r = 0.3
alpha_list = [0.01, 0.1, 0.3, 0.5, 1, 1.5, 1.8, 2]
k_end = 4
num_coh = 1
date = '22-02-2022'

results_dict = {}

for alpha in alpha_list:

    file_name_header = r'../Results/k_approx_error/M={}_N={}_r={}_alpha={}'.format(M, N, r, alpha)

    for file in glob.glob(file_name_header +
                           r'/num_coh={}_k=0-{}_{}*.csv'.format(num_coh, k_end, date)):

        results_dict[alpha] = pd.read_csv(file)

#
# file_name_header = r'../Results/k_approx_error/M={}_N={}_r={}_alpha={}'.format(M, N,r,alpha)
#
# results_dict = {}
#
# for num_coh in num_coh_range:
#
#     file_name_body = r'/num_coh={}_k=0-{}_{}.csv'.format(num_coh, k_end, date)
#     results_df = pd.read_csv(file_name_header + file_name_body)
#
#     results_dict[num_coh] = results_df

plot_name_header = r'../Plots/k_approx_error'

# Plot probability errors
plt.figure(1)
for alpha in results_dict.keys():

    k_to_plot = results_dict[alpha]['k']
    prob_error_to_plot = results_dict[alpha]['prob_error']

    plt.plot(k_to_plot, prob_error_to_plot, label='alpha={}'.format(alpha))#

plt.xlabel('k')
plt.ylabel('exact_prob - k_prob')
# plt.yscale('log')
plt.title('Prob error for n={},r={},num_coh={}'.format(n_photon,r, num_coh))
plt.legend()

plt.savefig(plot_name_header + r'/Prob_err_N={}_r={}_num_coh={}.png'.format(N, r, num_coh))

# Plot k-prob
plt.figure(2)
for alpha in results_dict.keys():
    k_to_plot = results_dict[alpha]['k']
    k_prob_to_plot = results_dict[alpha]['prob_k_approx']
    plt.plot(k_to_plot, k_prob_to_plot, label='alpha={}'.format(alpha))
plt.xlabel('k')
plt.ylabel('k_prob')
plt.title('k-th approx prob for n={},r={},num_coh={}'.format(n_photon,r, num_coh))
plt.legend()

plt.savefig(plot_name_header + r'/k_prob_N={}_r={}_num_coh={}.png'.format(N, r, num_coh))

#
# # Plot relative probability errors
# plt.figure(2)
# for num_coh in num_coh_range:
#
#     k_to_plot = results_dict[num_coh]['k']
#     prob_error_to_plot = results_dict[num_coh]['prob_error']
#     exact_prob_to_plot = results_dict[num_coh]['prob_exact']
#
#     relative_prob_error = prob_error_to_plot/ exact_prob_to_plot
#
#     plt.plot(k_to_plot, relative_prob_error, label='num_coh={}'.format(num_coh))#
#
# plt.xlabel('k')
# plt.ylabel('(exact_prob - k_prob)/ exact_prob')
# plt.title('Rel prob error for M={},N={},r={},alpha={}'.format(M,N,r,alpha))
# plt.legend()
#
# plt.savefig(plot_name_header + r'/Rel_prob_err_M={}_N={}_r={}_alpha={}.png'.format(M, N,r,alpha))
#

plt.show()