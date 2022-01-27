import pandas as pd

file_name = r'..\Results\test_pandas\test.csv'
results_df = pd.DataFrame(columns=['iteration'])

for i in range(10):
    results_df.loc[i] = {
        'iteration': i
    }
    results_df.to_csv(file_name)
