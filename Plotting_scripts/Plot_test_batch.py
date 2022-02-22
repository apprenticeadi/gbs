import matplotlib.pyplot as plt
import pandas as pd

df_nonbatch = pd.read_csv(r'../Results/k_approx_error/M=3_N=4_r=1.55_alpha=2.25/num_coh=1_k=0-4_15-02-2022.csv')
df_batch = pd.read_csv(r'../Results/k_approx_error/M=3_N=4_r=1.55_alpha=2.25/num_coh=1_k=0-4_batchTrue_17-02-2022.csv')

plt.figure(0)
plt.plot(df_nonbatch['k'], df_nonbatch['prob_exact'], label='exact')
plt.plot(df_nonbatch['k'], df_nonbatch['prob_k_approx'], label='non-batch')
plt.plot(df_batch['k'], df_batch['prob_k_approx'], label='batch')
plt.xlabel('k')
plt.ylabel('prob')
plt.legend()

for k in df_nonbatch['k']:
    print('k={}'.format(k))
    print('batch-nonbatch={}'
           .format(df_nonbatch['prob_k_approx'][k] - df_batch['prob_k_approx'][k]))
    print('exact-batch={}'
        .format(df_nonbatch['prob_exact'][k] - df_batch['prob_k_approx'][k]))
    print('exact-nonbatch={}'
          .format(df_nonbatch['prob_exact'][k] - df_nonbatch['prob_k_approx'][k]))


plt.figure(1)
plt.plot(df_nonbatch['k'], df_nonbatch['exact_time'], label='exact')
plt.plot(df_nonbatch['k'], df_nonbatch['k_time'], label='non-batch')
plt.plot(df_batch['k'], df_batch['k_time'], label='batch')
plt.xlabel('k')
plt.ylabel('time')
plt.legend()
plt.show()