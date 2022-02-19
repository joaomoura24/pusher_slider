import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_names = glob.glob("tracking_with_noise_*.csv")
num_files = len(file_names)
data = np.empty((num_files, 4))
# get names of *.csv files
# for file_name in glob.glob("*.csv"):
for i in range(num_files):
    file_name = file_names[i]
    # get dist level from file name
    # ang_dist = float(re.findall('([0-9]\.[0-9]*)', file_name)[0])
    ang_dist = float(file_name.split('_')[3])
    # create pandas DataFrame from csv
    df_in = pd.read_csv(file_name)
    x_nom = df_in["x_nom"].to_numpy()
    y_nom = df_in["y_nom"].to_numpy()
    x_opt = df_in["x_opt"].to_numpy()
    y_opt = df_in["y_opt"].to_numpy()
    # compute error percentiles
    err = np.sqrt((x_nom - x_opt)**2 + (y_nom - y_opt)**2)
    median = np.percentile(err, 50)
    lower = np.percentile(err, 10)
    upper = np.percentile(err, 90)
    # save data
    data_i = np.array([ang_dist, median, lower, upper]).transpose()
    data[i, :] = data_i

# sort data
column_names = ['ang_dist', 'err_median', 'err_lower', 'err_upper']
df = pd.DataFrame(data=data,
        columns=column_names)
df_out = df.sort_values(by=['ang_dist'])

# plot data
ax = plt.subplot()
ax.fill_between(df_out['ang_dist'], df_out['err_lower'], df_out['err_upper'],
        facecolor='blue', alpha=0.2)
ax.plot(df_out['ang_dist'], df_out['err_median'], color='blue')
plt.show()

# save data
df_out.to_csv("error_with_dist_" + file_name.split('_')[4],
        float_format='%.5f')
