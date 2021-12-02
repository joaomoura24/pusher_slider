import os
import _pickle as pickle
import datetime
import numpy as np


# User input
tunnel_center = np.array([-1, 0.8])
tunnel_width = 1.1
start = np.array([1.6, 0.15])
goal = tunnel_center + np.array([0, 0.9])
obs_radius = 0.05

# Create wall
nseg1 = 5
seg1x = tunnel_center[0] - tunnel_width/2.0 + np.linspace(-0.8, 0, nseg1)
seg1y = tunnel_center[1]*np.ones(nseg1)

nseg2 = 5
seg2x = (tunnel_center[0] - tunnel_width/2.0)*np.ones(nseg2)
seg2y = tunnel_center[1] + np.linspace(0, 1.4, nseg2)

nseg3 = 5
seg3x = tunnel_center[0] + np.linspace(-tunnel_width/2.0, tunnel_width/2.0, nseg3)
seg3y = tunnel_center[1] + 1.4*np.ones(nseg3)

nseg4 = 5
seg4x = (tunnel_center[0] + tunnel_width/2.0)*np.ones(nseg4)
seg4y = tunnel_center[1] + np.linspace(0, 1.4, nseg4)

nseg5 = 10
seg5x = tunnel_center[0] + tunnel_width/2.0 + np.linspace(0, 3, nseg5)
seg5y = tunnel_center[1]*np.ones(nseg5)

# Parse obs
obs = []
for k in range(nseg1-1):
    obs.append((seg1x[k], seg1y[k], obs_radius))
for k in range(nseg2-1):
    obs.append((seg2x[k], seg2y[k], obs_radius))
for k in range(nseg3-1):
    obs.append((seg3x[k], seg3y[k], obs_radius))
for k in range(1, nseg4):
    obs.append((seg4x[k], seg4y[k], obs_radius))
for k in range(nseg5):
    obs.append((seg5x[k], seg5y[k], obs_radius))

# Save data
data = {'obs': obs, 'start': start, 'goal': goal}
timestamp = datetime.datetime.now()
filename = 'env_tunnel_use.dat'
filename_full = os.path.join(os.getcwd(), 'environment_descriptions', filename)
with open(filename_full, 'wb') as f:
    pickle.dump(data, f)
print("Saved", filename, 'in environment_descriptions/')
