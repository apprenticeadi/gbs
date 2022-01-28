import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")

start_M = 2
end_M = 8
M_list = list(range(start_M, end_M + 1))

r=1.55
alpha = 4

num_M = len(M_list)

time_coh_0 = np.zeros(num_M)
time_coh_1 = np.zeros(num_M)
time_coh_half = np.zeros(num_M)
time_coh_but_one = np.zeros(num_M)
time_coh_all = np.zeros(num_M)

time_dict = {
    'zero': time_coh_0,
    'one': time_coh_1,
    'half+1': time_coh_half,
    'all but one': time_coh_but_one,
    'all': time_coh_all
}

file_name_start = r'..\Results\varying_coh\M='
file_name_end_1 = r'r={}_alpha={}_26-01-2022.csv'.format(r, alpha)
file_name_end_2 = r'r={}_alpha={}_27-01-2022.csv'.format(r, alpha)

iteration = 0
for M in M_list:

    num_coh_dict = {
        'zero': 0,
        'one': 1,
        'half+1': int(M/2)+1,
        'all but one': M-1,
        'all': M
    }

    for num_coh in time_dict.keys():
        num_coh_number = num_coh_dict[num_coh]
        df = pd.read_csv(file_name_start
                         + '{}_coh={}_'.format(M, num_coh_number)
                         + file_name_end_2)

        time_dict[num_coh][iteration] = sum(df['time'])

    iteration += 1

plt.figure(0)
for num_coh in time_dict.keys():

    plt.plot(M_list, time_dict[num_coh], label='{} coherent states'.format(num_coh))

plt.xlabel('Number of modes M')
plt.ylabel('Time(s)')
plt.title('Run time for 100 samples, r={}, alpha={}'.format(r, alpha))
plt.legend()
plt.xticks(range(start_M, end_M + 1))
plt.show()

plot_filename = r'..\Plots\varying_coh_M={}-{}_r={}_alpha={}_{}.png'.format(start_M, end_M, r, alpha, date_stamp)
plt.savefig(plot_filename)