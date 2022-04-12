import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data for plotting
file_mi = "mi/error_with_dist_mi.csv"
df_mi = pd.read_csv(file_mi)
file_cc = "cc/error_with_dist_cc.csv"
df_cc = pd.read_csv(file_cc)

# plot data
ax = plt.subplot()
# mi
ax.fill_between(df_mi['ang_dist'], df_mi['err_lower'], df_mi['err_upper'],
        facecolor='blue', alpha=0.2)
ax.plot(df_mi['ang_dist'], df_mi['err_median'], color='blue', label='mi')
# cc
ax.fill_between(df_cc['ang_dist'], df_cc['err_lower'], df_cc['err_upper'],
        facecolor='red', alpha=0.2)
ax.plot(df_cc['ang_dist'], df_cc['err_median'], color='red', label='cc')
# ax.set_yscale('log')
# plot
ax.legend()
plt.show()
