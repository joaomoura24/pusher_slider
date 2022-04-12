import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_names_state = glob.glob("tracking_time_horizon_*_state.csv")
file_names_time = glob.glob("tracking_time_horizon_*_time.csv")
num_files = len(file_names_state)
data_state = np.empty((num_files, 4))
data_time = np.empty((num_files, 4))
for i in range(num_files):
    file_name_state = file_names_state[i]
    file_name_time = file_names_time[i]
    # get dist level from file name
    th_state = int(file_name_state.split('_')[3])
    th_time = int(file_name_time.split('_')[3])
    # create pandas DataFrame from csv to state
    df_state = pd.read_csv(file_name_state)
    x_nom = df_state["x_nom"].to_numpy()
    y_nom = df_state["y_nom"].to_numpy()
    x_opt = df_state["x_opt"].to_numpy()
    y_opt = df_state["y_opt"].to_numpy()
    # create pandas DataFrame from csv to time
    df_time = pd.read_csv(file_name_time)
    comp_time = df_time["comp_time"].to_numpy()
    # compute error percentiles - time
    comp_time_median = np.percentile(comp_time, 50)*1000
    comp_time_lower = np.percentile(comp_time, 10)*1000
    comp_time_upper = np.percentile(comp_time, 90)*1000
    # compute error percentiles - error
    err = np.sqrt((x_nom - x_opt)**2 + (y_nom - y_opt)**2)
    err_median = np.percentile(err, 50)*1000
    err_lower = np.percentile(err, 10)*1000
    err_upper = np.percentile(err, 90)*1000
    # save data
    data_state[i, :] = np.array(
            [th_state, err_median, err_lower, err_upper]).transpose()
    data_time[i, :] = np.array(
            [th_time, comp_time_median, comp_time_lower, comp_time_upper]).transpose()

# sort data
column_names_state = ['th', 'err_median', 'err_lower', 'err_upper']
column_names_time = ['th', 'comp_time_median', 'comp_time_lower', 'comp_time_upper']
df_state = pd.DataFrame(data=data_state,
        columns=column_names_state)
df_time = pd.DataFrame(data=data_time,
        columns=column_names_time)
df_out_state = df_state.sort_values(by=['th'])
df_out_time = df_time.sort_values(by=['th'])
# df_out = pd.concat([df_out_state, df_out_time], axis=1, sort=False)
df_out = pd.merge(df_out_state, df_out_time, on=['th'])
df_out = df_out.reset_index(drop=True)

# plot data
ax = plt.subplot()
ax.fill_between(df_out['th'], df_out['comp_time_lower'], df_out['comp_time_upper'],
        facecolor='blue', alpha=0.2)
ax.plot(df_out['th'], df_out['comp_time_median'], color='blue')
plt.show()
# another plot
ax2 = plt.subplot()
ax2.fill_between(df_out['th'], df_out['err_lower'], df_out['err_upper'],
        facecolor='blue', alpha=0.2)
ax2.plot(df_out['th'], df_out['err_median'], color='blue')
plt.show()

# save data
df_out.to_csv("time_horizon_comparison_" + file_name_state.split('_')[4] + ".csv",
        float_format='%.5f')
