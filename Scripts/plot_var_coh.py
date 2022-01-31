import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

sub_sample_size = 100
repeat_num = 40
plot_num = 1
r = 1.55
alpha = 2
M_start = 2
M_end = 9
M_list = list(range(M_start, M_end + 1))
num_M = len(M_list)
date_stamp = '28-01-2022'

file_head = r'..\Results\varying_coh_{}x{}_samples'.format(sub_sample_size, repeat_num)

plot_dict = {
    'No': np.zeros(num_M),
    'One': np.zeros(num_M),
    'Half': np.zeros(num_M),
    'All but one': np.zeros(num_M),
    'All': np.zeros(num_M),
}

iteration_M = 0
for M in range(M_start, M_end + 1):

    coh_num_dict = {
        'No': 0,
        'One': 1,
        'Half': int(M / 2 + 0.5),
        'All but one': M - 1,
        'All': M,
    }

    for coh_num_key in coh_num_dict.keys():

        coh_num = coh_num_dict[coh_num_key]

        file_head_2 = r'\M={}_coh={}_r={}_alpha={}'.format(M, coh_num, r, alpha)

        time_sum = 0

        for i in range(plot_num):
            file_body = r'\No_{}_{}.csv'.format(i, date_stamp)

            results_df = pd.read_csv(file_head +
                                     file_head_2 +
                                     file_body
                                     )

            time_sum += sum(results_df['time'])

        plot_dict[coh_num_key][iteration_M] = time_sum

    iteration_M += 1

plt.figure(0)
for coh_num_key in coh_num_dict.keys():
    plt.plot(M_list, plot_dict[coh_num_key], label='{} coherent states'.format(coh_num_key))

plt.xlabel('Number of modes M')
plt.ylabel('Time(s)')
plt.title('Run time for {}x{} samples, r={}, alpha={}'.format(sub_sample_size, repeat_num, r, alpha))
plt.legend()
plt.xticks(range(M_start, M_end + 1))
plt.show()

plot_filename = r'..\Plots\var_coh_M={}-{}_r={}_alpha={}_{}x{}_{}.png'.format(M_start, M_end, r, alpha,
                                                                              sub_sample_size, plot_num,
                                                                              date_stamp)
plt.savefig(plot_filename)

