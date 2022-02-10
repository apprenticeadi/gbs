import matplotlib.pyplot as plt
import pandas as pd

M=8
num_coh = 4
r=1.55
alpha=2
date_stamp = '09-02-2022'

results_df = pd.read_csv(r'..\Results\test_chain_k_approx_timed\M={}_num_coh={}_r={}_alpha={}_{}.csv'
                         .format(M, num_coh, r, alpha, date_stamp))

num_rows = results_df.shape[0]
label_list = list(range(num_rows))

plt.figure(0)
plt.plot(label_list, results_df['Jake_time'], label='Jake_time')
plt.plot(label_list, results_df['k_time'], label='k_time')

plt.xlabel('Chain rule iteration')
plt.ylabel('Runtime')
plt.legend()
plt.title('Run time comparison')
plt.show()