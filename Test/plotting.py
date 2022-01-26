import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M_list = [2, 3, 4, 5, 6, 7, 8]

num_M = len(M_list)

time_coh_0 = np.zeros(num_M)
time_coh_1 = np.zeros(num_M)
time_coh_all = np.zeros(num_M)

time_dict = {
    0: time_coh_0,
    1: time_coh_1,
    -1: time_coh_all
}

file_name_start = r'..\Results\M='
file_name_end = r'r=0.3_alpha=0.6_26-01-2022.csv'

iteration = 0
for M in M_list:
    for num_coh in time_dict.keys():
        if num_coh == -1:
            df = pd.read_csv(file_name_start +
                           '{}_coh={}_'.format(M, M) +
                           file_name_end)

        else:
            df = pd.read_csv(file_name_start +
                             '{}_coh={}_'.format(M, num_coh) +
                             file_name_end)

        time_dict[num_coh][iteration] = sum(df['time'])

    iteration += 1


plt.figure(0)
for num_coh in time_dict.keys():

    if num_coh==-1:
        plt.plot(M_list, time_dict[num_coh], label='All coherent states')
    else:
        plt.plot(M_list, time_dict[num_coh], label='{} coherent states'.format(num_coh))

plt.xlabel('Number of modes M')
plt.ylabel('Time(s)')
plt.title('Run time for generating 100 samples')
plt.legend()
plt.show()






