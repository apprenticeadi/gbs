import matplotlib.pyplot as plt
import pandas as pd

M = 8
N = 16
num_coh_range = list(range(1, 8))

file_name_header = r'../Results/k_approx_error/M={}_N={}'.format(M, N)

date_1 = '10-02-2022'
date_2 = '11-02-2022'

results_dict = {}

for num_coh in num_coh_range:

    file_name_body = r'/num_coh={}_'.format(num_coh)

    try:
        results_df = pd.read_csv(file_name_header
                                 + file_name_body
                                 + '{}.csv'.format(date_1))
    except FileNotFoundError:
        results_df = pd.read_csv(file_name_header
                                 + file_name_body
                                 + '{}.csv'.format(date_2))

    try:
        results_df_2 = pd.read_csv(file_name_header
                                   + file_name_body
                                   + r'k=4-5_'
                                   + '{}.csv'.format(date_2))

        results_df = pd.concat([results_df, results_df_2])
    except FileNotFoundError:
        pass

    results_dict[num_coh] = results_df

plot_name_header = r'../Plots/k_approx_error'

# # Plot run time for 1 coherent state
# time_1_coh = results_dict[1]['k_time']
# k_array_1_coh = results_dict[1]['k']

# plt.figure(0)
# plt.plot(k_array_1_coh, time_1_coh, 'x')
# plt.xlabel('k')
# plt.ylabel('runtime (s)')
# plt.title('k-th order approximation run time for num_coh=1')
# plt.yscale('log')
#
# plt.savefig(plot_name_header + r'/Runtime_M={}_N={}_num_coh=1'.format(M, N))

# Plot probability errors
plt.figure(1)
for num_coh in num_coh_range:

    k_to_plot = results_dict[num_coh]['k']
    prob_error_to_plot = results_dict[num_coh]['prob_error']

    plt.plot(k_to_plot, prob_error_to_plot, label='num_coh={}'.format(num_coh))#

plt.xlabel('k')
plt.ylabel('exact_prob - k_prob')
plt.yscale('log')
plt.title('Probability error for k-th order approximation')
plt.legend()

plt.savefig(plot_name_header + r'/Prob_error_M={}_N={}.png'.format(M, N))


# Plot relative probability errors
plt.figure(2)
for num_coh in num_coh_range:

    k_to_plot = results_dict[num_coh]['k']
    prob_error_to_plot = results_dict[num_coh]['prob_error']
    exact_prob_to_plot = results_dict[num_coh]['prob_exact']

    relative_prob_error = prob_error_to_plot/ exact_prob_to_plot

    plt.plot(k_to_plot, relative_prob_error, label='num_coh={}'.format(num_coh))#

plt.xlabel('k')
plt.ylabel('(exact_prob - k_prob)/ exact_prob')
plt.title('Relative probability error for k-th order approximation')
plt.legend()

plt.savefig(plot_name_header + r'/Rel_prob_error_M={}_N={}.png'.format(M, N))


plt.show()