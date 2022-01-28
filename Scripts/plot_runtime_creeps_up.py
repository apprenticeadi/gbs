import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

date_stamp = datetime.datetime.now().strftime("%d-%m-%Y")

M = 2
r=1.55
alpha = 0.6
file_date = '28-01-2022'
num_sample = 4096
cut_off = 4096

file_name_head = r'..\Results\varying_coh_4096_samples\test\testM={}_'.format(M)
file_name_tail = r'r={}_alpha={}_{}.csv'.format(r, alpha, file_date)

num_coh_dict = {
    'zero': 0,
    # 'one': 1,
    'half': int(M / 2+0.5),
    # 'all but one': M - 1,
    'all': M
}

num_coh_keys = [
    'zero',
    # 'one',
    'half',
    # 'all but one',
    'all'
]

time_dict = dict.fromkeys(num_coh_keys)

plt.figure(0)

for num_coh in num_coh_keys:
    file_name_body = r'coh={}_'.format(num_coh_dict[num_coh])
    df = pd.read_csv(
        file_name_head + file_name_body + file_name_tail
    )
    assert len(df['time']) == num_sample
    total_time = sum(df['time'])
    time_dict[num_coh] = total_time

    plt.plot(list(range(num_sample))[:cut_off], df['time'][:cut_off], label='{} coherent'.format(num_coh))

    print('{} coherent total run time = {}'.format(num_coh, total_time))

plt.xlabel('#Sample')
plt.ylabel('Time(s)')
plt.title('Sampling time for {} samples, M={}, r={}, alpha={}'.format(num_sample, M, r, alpha))
plt.legend()
plt.show()

if cut_off == num_sample:
    plot_filename = r'..\Plots\runtime_creeps_up_{}_samples_M={}_r={}_alpha={}_{}.pdf'.format(num_sample, M, r, alpha, date_stamp)
    plt.savefig(plot_filename)


