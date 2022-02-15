import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
import os

M = 3
N = 4
r=1.55
alpha=2.25
k_end = 4
num_coh_range = [1]
time_stamp = datetime.datetime.now().strftime("%d-%m-%Y(%H-%M-%S.%f)")

file_name_header = r'../Results/k_approx_error/M={}_N={}_r={}_alpha={}'.format(M, N,r,alpha)

date = '15-02-2022'

results_dict = {}

for num_coh in num_coh_range:

    # Still under construction, don't know how to compare two strings partially
    # for file_name in os.listdir(file_name_header):
    #     if re.match(r'num_coh={}_k=0-{}_{}.csv'.format(num_coh, k_end, date), file_name):
    #         results_df = pd.read_csv(file_name)
    #     else:
    #         raise Exception('File not found')

    results_df = pd.read_csv(file_name_header + r'/num_coh={}_k=0-{}_{}.csv'.format(num_coh, k_end, date))
    # results_df = pd.read_csv(file_name_header + r'/num_coh=1_k=0-4_15-02-2022(17-58-23.765881).csv')
    results_dict[num_coh] = results_df

plot_name_header = r'../Plots/k_approx_error'
#
# # Plot probability errors
# plt.figure(1)
# for num_coh in num_coh_range:
#
#     k_to_plot = results_dict[num_coh]['k']
#     prob_error_to_plot = results_dict[num_coh]['prob_error']
#
#     plt.plot(k_to_plot, prob_error_to_plot, label='num_coh={}'.format(num_coh))#
#
# plt.xlabel('k')
# plt.ylabel('exact_prob - k_prob')
# plt.yscale('log')
# plt.title('Prob error for M={},N={},r={},alpha={}'.format(M,N,r,alpha))
# plt.legend()
#
# # plt.savefig(plot_name_header + r'/Prob_err_M={}_N={}_r={}_alpha={}_{}.png'.format(M, N, r, alpha, time_stamp))
#
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

# plt.savefig(plot_name_header + r'/Rel_prob_err_M={}_N={}_r={}_alpha={}.png'.format(M, N,r,alpha, time_stamp))


# Plot probability
plt.figure(3)
for num_coh in num_coh_range:

    k_to_plot = results_dict[num_coh]['k']

    plt.plot(k_to_plot, results_dict[num_coh]['prob_k_approx'],
             label='num_coh={}, approx'.format(num_coh))
    plt.plot(k_to_plot, results_dict[num_coh]['prob_exact'],
             label='num={}, exact'.format(num_coh))

plt.xlabel('k')
plt.ylabel('prob')
plt.title('Prob for M={},N={},r={},alpha={}'.format(M,N,r,alpha))
plt.legend()

# plt.savefig(plot_name_header + r'/Prob_M={}_N={}_r={}_alpha={}_{}.png'.format(M, N,r,alpha, time_stamp))



plt.show()