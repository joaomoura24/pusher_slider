import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_names = glob.glob("tracking_time_horizon_*.csv")
num_files = len(file_names)
data = np.empty((num_files, 4))
# get names of *.csv files
# for file_name in glob.glob("*.csv"):
for i in range(num_files):
    file_name = file_names[i]
    print('****************')
    print(file_name)
    # get dist level from file name
    th = int(file_name.split('_')[3])
    print(th)
    pass
    # create pandas DataFrame from csv
    df_in = pd.read_csv(file_name)
    comp_time = df_in["comp_time"].to_numpy()
    # compute error percentiles
    median = np.percentile(comp_time, 50)
    lower = np.percentile(comp_time, 10)
    upper = np.percentile(comp_time, 90)
    # save data
    data_i = np.array([th, median, lower, upper]).transpose()
    data[i, :] = data_i

# sort data
column_names = ['th', 'comp_time_median', 'comp_time_lower', 'comp_time_upper']
df = pd.DataFrame(data=data,
        columns=column_names)
df_out = df.sort_values(by=['th'])

# plot data
ax = plt.subplot()
ax.fill_between(df_out['th'], df_out['comp_time_lower'], df_out['comp_time_upper'],
        facecolor='blue', alpha=0.2)
ax.plot(df_out['th'], df_out['comp_time_median'], color='blue')
plt.show()

# save data
df_out.to_csv("comp_time_with_diff_th_" + file_name.split('_')[4],
        float_format='%.5f')
