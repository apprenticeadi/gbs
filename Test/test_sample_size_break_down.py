import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


file_name_header = r'..\Results\varying_coh_100x15_samples\test\M=2_coh=0_r=1.55_alpha=0.6_'
file_name_tail = r'_28-01-2022.csv'

for i in range(15):
    df = pd.read_csv(file_name_header + '{}'.format(i) + file_name_tail)
    run_time = sum(df['time'])

    print('{} total time = {}'.format(i, run_time))
