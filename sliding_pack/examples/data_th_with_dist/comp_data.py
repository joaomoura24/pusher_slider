import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data for plotting
file_mi = "mi/time_horizon_comparison_mi.csv"
df_mi = pd.read_csv(file_mi)
file_cc = "cc/time_horizon_comparison_cc.csv"
df_cc = pd.read_csv(file_cc)

# plot data
ax = plt.subplot()
# mi
ax.fill_between(df_mi['th'], df_mi['comp_time_lower'], df_mi['comp_time_upper'],
        facecolor='blue', alpha=0.2)
ax.plot(df_mi['th'], df_mi['comp_time_median'], color='blue', label='mi')
# cc
ax.fill_between(df_cc['th'], df_cc['comp_time_lower'], df_cc['comp_time_upper'],
        facecolor='red', alpha=0.2)
ax.plot(df_cc['th'], df_cc['comp_time_median'], color='red', label='cc')
plt.title(label='comp time')
# ax.set_yscale('log')
# plot
ax.legend()
plt.show()

# plot data
ax = plt.subplot()
# mi
ax.fill_between(df_mi['th'], df_mi['err_lower'], df_mi['err_upper'],
        facecolor='blue', alpha=0.2)
ax.plot(df_mi['th'], df_mi['err_median'], color='blue', label='mi')
# cc
ax.fill_between(df_cc['th'], df_cc['err_lower'], df_cc['err_upper'],
        facecolor='red', alpha=0.2)
ax.plot(df_cc['th'], df_cc['err_median'], color='red', label='cc')
plt.title(label='error')
# ax.set_yscale('log')
# plot
ax.legend()
plt.show()
