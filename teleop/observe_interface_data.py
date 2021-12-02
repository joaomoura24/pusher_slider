import os
import sys
import numpy as np
from pprint import pprint
import _pickle as pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

R90 = np.array([[0, -1], [1, 0]])  # rotation matrix about 90 degrees

def pprint_dict(d, indent=0):
    indent_spaces = " "*indent
    print(indent_spaces, '{')
    for k, v in d.items():
        print(indent_spaces, "'%s'"%k, ':', end='')
        if isinstance(v, dict):
            print('')
            pprint_dict(v, indent=indent+4)
        elif isinstance(v, np.ndarray):
            print(type(v), "shape =", v.shape)
        elif isinstance(v, (tuple, list)):
            first_type = type(v[0])
            if all([first_type == type(x) for x in v]):
                print(type(v), "elements have type", first_type, "len =", len(v))
            else:
                print(type(v), "len =", len(v), 'note elements have mixed types')
        else:
            print(type(v))
    print(indent_spaces, '}')


class InterfaceData:

    def __init__(self, interface_data_filename):

        # Load data
        with open(interface_data_filename, 'rb') as interface_data_file:
            interface_data = pickle.load(interface_data_file)
            print("Loaded interface data from", interface_data_filename)
            pprint_dict(interface_data)

        # Extract data
        x = np.array(interface_data['x'], dtype=float)  # states (see below)
        self.t = np.array(interface_data['t'], dtype=float)  # time
        self.t -= self.t[0]  # ensure time starts from 0
        self.cF = np.array(interface_data['cf'], dtype=bool)  # contactflags
        self.obs_p = np.array([e[:2] for e in interface_data['env']['obs']], dtype=float).T  # positions of obstacles in world frame
        self.obs_r = np.array([e[2] for e in interface_data['env']['obs']], dtype=float)  # radii of obstacles
        self.start = np.array(interface_data['env']['start'])  # start position
        self.goal = np.array(interface_data['env']['goal'])  # goal position
        self.u_pusher = np.array(interface_data['u_pusher'])  # human input: contact forces (normal and tangential)
        self.u_point = np.array(interface_data['u_point'])  # human input: velocity of pusher in world frame
        self.Nobs = self.obs_r.shape[0]  # number of obstacles

        # Split data
        self.xB = x[0,:]  # x-position of body in world frame
        self.yB = x[1,:]  # y-position of body in world frame
        self.thetaB = x[2,:]  # heading of body in world frame
        self.phiP = x[3,:]  # angle of pusher in body frame
        self.xP = x[4,:]  # x-position of pusher in body frame
        self.yP = x[5,:]  # y-position of pusher in body frame

        # Compute r
        self.rP = np.zeros(self.t.shape)  # (rP, phiP) polar coordinate of pusher in body frame
        for k in range(self.t.shape[0]):
            self.rP[k] = np.sqrt(self.xP[k]**2 + self.yP[k]**2)

        # Compute derivates
        self.dxB = np.gradient(self.xB, self.t) # x-velocity of body in world frame
        self.dyB = np.gradient(self.yB, self.t) # y-velocity of body in world frame
        self.dthetaB = np.gradient(self.thetaB, self.t) # angular velocity of body in world frame
        self.dphiP = np.gradient(self.phiP, self.t)  # angular velocity of pusher in body frame
        self.dxP = np.gradient(self.xP, self.t) # x-velocity of pusher in body frame
        self.dyP = np.gradient(self.yP, self.t) # y-velocity of pusher in body frame

        # Compute xPW, xPW
        self.xPW = np.zeros(self.xP.shape)  # x-positions of pusher in world frame
        self.yPW = np.zeros(self.yP.shape)  # y-positions of pusher in world frame
        for k in range(self.xPW.shape[0]):

            c, s = np.cos(self.thetaB[k]), np.sin(self.thetaB[k])
            body_R = np.array([[c, -s], [s, c]])
            body_p = np.array([self.xB[k], self.yB[k]])
            pusher_p = np.array([self.xP[k], self.yP[k]])
            pusher_p_W = body_R@pusher_p + body_p

            self.xPW[k] = pusher_p_W[0]
            self.yPW[k] = pusher_p_W[1]

        # Compute dxPW, dyPW
        self.dxPW = np.gradient(self.xPW, self.t) # x-velocity of pusher in world frame
        self.dyPW = np.gradient(self.yPW, self.t) # y-velocity of pusher in world frame

        # Interpolate
        self.xB_fun = interp1d(self.t, self.xB)
        self.yB_fun = interp1d(self.t, self.yB)
        self.thetaB_fun = interp1d(self.t, self.thetaB)
        # self.cF_fun ... see below
        self.phiP_fun = interp1d(self.t, self.phiP)
        self.xP_fun = interp1d(self.t, self.xP)
        self.yP_fun = interp1d(self.t, self.yP)
        self.xPW_fun = interp1d(self.t, self.xPW)
        self.yPW_fun = interp1d(self.t, self.yPW)

        # Compute time segments
        start_t = 0.0
        prev_cF = True
        in_contact = True
        time_segs = []
        for k in range(self.t.shape[0]):

            curr_t = self.t[k]
            curr_cF = self.cF[k]

            if curr_cF != prev_cF:
                # True -> switch

                # Save end t
                end_t = curr_t

                # Save
                time_segs.append(dict(segment_start_time=start_t, segment_end_time=end_t, in_contact=in_contact))
                in_contact = not in_contact

                # Reset
                start_t = end_t

            prev_cF = curr_cF

        start_t = time_segs[-1]['segment_end_time']
        end_t = self.t[-1]
        time_segs.append(dict(segment_start_time=start_t, segment_end_time=end_t, in_contact=in_contact))
        self.time_segments = time_segs
        # pprint(time_segs)

    def cF_fun(self, t):
        if isinstance(t, (np.ndarray, list, tuple)):
            t = np.asarray(t)
            cF_out = np.zeros(t.shape[0], dtype=bool)
            for k in range(t.shape[0]):
                idx = np.argmin(abs(self.t - t[k]))
                cF_out[k] = self.cF[idx]
            ret = cF_out
        elif isinstance(t, (float, int, long)):
            idx = np.argmin(abs(self.t - t))
            ret = self.cF[idx]
        return ret

    def downsample_environment(self, n_keep, random_seed=None):
        if isinstance(random_seed, int):
            np.random.seed(random_seed)
        idx = np.arange(self.Nobs)  # indices of obstacles
        np.random.shuffle(idx)   # shuffle indices
        idx = idx[:n_keep]  # extract first n_keep_in_downsample, this
                            # represents a random sample of
                            # n_keep_in_downsample obstacles from full
                            # environment model
        Nobs_full = self.Nobs
        self.obs_p = self.obs_p[:, idx]
        self.obs_r = self.obs_r[idx]
        self.Nobs = self.obs_r.shape[0]  # reset Nobs
        print(f"Environment model reduced from {Nobs_full} points to {self.Nobs}.")



class DataPlotter:

    def __init__(self, interface_data):
        self.interface_data = interface_data
        self.ax_birdseye = None
        self.ax_panel = None

    def plot_birdseye(self):

        # Plot birds eye view
        fig, ax = plt.subplots(tight_layout=True, figsize=(12, 12))

        ax.plot(self.interface_data.xB, self.interface_data.yB, '-g', label='Body path')
        # ax.plot(self.interface_data.xPW, self.interface_data.yPW, '-b', label='Pusher path')

        is_first_r = True
        is_first_b = True
        for seg in self.interface_data.time_segments:
            start_time = seg['segment_start_time']
            end_time = seg['segment_end_time']
            t = np.linspace(start_time, end_time, 50)
            xx = self.interface_data.xPW_fun(t)
            yy = self.interface_data.yPW_fun(t)

            if seg['in_contact']:
                fmt = '-'
            else:
                fmt = ':'

            add = dict()

            if seg['in_contact'] and is_first_r:
                add['label'] = 'pusher in contact'
                is_first_r = False
            if (not seg['in_contact']) and is_first_b:
                add['label'] = 'pusher not in contact'
                is_first_b = False

            ax.plot(xx, yy, fmt+'b', **add)

        # is_first_r = True
        # is_first_b = True

        # for k in range(self.interface_data.t.shape[0]):

        #     if self.interface_data.cF[k]:
        #         color='r'
        #         label = 'pusher in contact'
        #         in_contact = True
        #     else:
        #         color='b'
        #         label = 'pusher not in contact'
        #         in_contact = False


        #     plt_config = dict(color=color, markersize=0.5)
        #     if in_contact and is_first_r:
        #         plt_config['label'] = label
        #         is_first_r = False
        #     if (not in_contact) and is_first_b:
        #         plt_config['label'] = label
        #         is_first_b = False

        #     ax.plot(self.interface_data.xPW[k], self.interface_data.yPW[k], 'o', **plt_config)


        # for s, e in time_segs:
        #     t = np.linspace(s, e, 100000)
        #     ax.plot(self.interface_data.xPW_fun(t), self.interface_data.yPW_fun(t), '-r')

        # Plot rectangles to represent body
        n_rect = 30
        alpha = np.linspace(0.01, 0.5, n_rect)
        t = np.linspace(0, self.interface_data.t.max(), n_rect)
        for k in range(n_rect):
            tt = t[k]
            xy_rect_mid = np.array([self.interface_data.xB_fun(tt), self.interface_data.yB_fun(tt)])
            angle_rect = self.interface_data.thetaB_fun(tt)
            dr_x = np.array([np.cos(angle_rect), np.sin(angle_rect)])
            xy_rect_corner = xy_rect_mid + (-.25*dr_x) + (-.25*R90@dr_x)
            rect_input = dict(angle=np.rad2deg(angle_rect), color='#BDB76B', alpha=alpha[k])
            if k == n_rect-1:
                rect_input['label'] = 'Body'
            ax.add_patch(mpl_patches.Rectangle(xy_rect_corner, 0.5, 0.5, **rect_input))

        # Plot circles to represent pusher
        n_circ = 150
        alpha = np.linspace(0.1, 0.5, n_circ)
        t = np.linspace(0, self.interface_data.t.max(), n_circ)
        for k in range(n_circ):
            tt = t[k]
            plt_input = dict(alpha=alpha[k])
            if k == n_circ-1:
                plt_input['label'] = 'pusher'
            ax.plot([self.interface_data.xPW_fun(tt)], [self.interface_data.yPW_fun(tt)], 'ob', **plt_input)


        ax.plot([self.interface_data.start[0]], [self.interface_data.start[1]], 'go', label='Start')
        ax.plot([self.interface_data.goal[0]], [self.interface_data.goal[1]], 'ro', label='Goal')

        for i in range(self.interface_data.Nobs):
            circle_input = dict(xy=self.interface_data.obs_p[:, i], radius=self.interface_data.obs_r[i], color='k')
            if i == 0:
                circle_input['label'] = 'Obstacle'
            ax.add_patch(mpl_patches.Circle(**circle_input))

        ds_plot_pad = 1
        xlim = [
            self.interface_data.obs_p[0,:].min()-self.interface_data.obs_r.min()-ds_plot_pad,
            self.interface_data.obs_p[0,:].max()+self.interface_data.obs_r.max()+ds_plot_pad,
        ]
        ylim = [
            self.interface_data.obs_p[1,:].min()-self.interface_data.obs_r.min()-ds_plot_pad,
            self.interface_data.obs_p[1,:].max()+self.interface_data.obs_r.max()+ds_plot_pad,
        ]
        ax.grid()
        ax.legend(loc='upper right')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')

        self.ax_birdseye = ax

        print("Completed birdseye view plot")

    def plot_panel(self):

        # Plot time evolutions
        fig, ax = plt.subplots(3, 2, tight_layout=True, sharex=True, figsize=(10, 8))

        for a in ax[-1, :]:
            a.set_xlabel('Time (s)')

        def plot_time(ax, t, data, colors, labels, ylabel):
            ax.set_ylabel(ylabel)
            if isinstance(data, list):
                for d, l, c in zip(data, labels, colors):
                    ax.plot(t, d, '-', color=c, label=l)
            else:
                ax.plot(t, data, '-', color=colors, label=labels)

        # body
        plot_time(ax[0, 0], self.interface_data.t, [self.interface_data.xB, self.interface_data.yB], ['r', 'b'], ['x', 'y'], 'Body position')
        plot_time(ax[1, 0], self.interface_data.t, [self.interface_data.dxB, self.interface_data.dyB], ['r', 'b'], ['x', 'y'], 'Body velocity')
        plot_time(ax[2, 0], self.interface_data.t, self.interface_data.thetaB, 'g', 'theta', 'Body heading')

        # pusher
        # plot_time(ax[0, 1], self.interface_data.t, [self.interface_data.xPW, self.interface_data.yPW], ['r', 'b'], ['x', 'y'], 'Pusher position')
        plot_time(ax[1, 1], self.interface_data.t, [self.interface_data.dxPW, self.interface_data.dyPW], ['r', 'b'], ['x', 'y'], 'Pusher velocity')
        plot_time(ax[2, 1], self.interface_data.t, self.interface_data.phiP, 'g', 'phi', 'Pusher angle (body frame)')

        is_first_r = True
        is_first_b = True
        for seg in self.interface_data.time_segments:

            start_time = seg['segment_start_time']
            end_time = seg['segment_end_time']
            t = np.linspace(start_time, end_time, 50)

            if seg['in_contact']:
                fmtx = '-'
                fmty = '-'
            else:
                fmtx = ':'
                fmty = ':'

            addx = dict()
            addy = dict()

            if seg['in_contact'] and is_first_r:
                addx['label'] = 'x-pos in contact'
                addy['label'] = 'y-pos in contact'
                is_first_r = False
            if (not seg['in_contact']) and is_first_b:
                addx['label'] = 'x-pos not in contact'
                addy['label'] = 'y-pos not in contact'
                is_first_b = False

            ax[0, 1].plot(t, self.interface_data.xPW_fun(t), fmtx, color='r', **addx)
            ax[0, 1].plot(t, self.interface_data.yPW_fun(t), fmty, color='b', **addy)

        ax[0, 1].set_ylabel('Pusher position')

        # is_first_r = True
        # is_first_b = True
        # for k in range(self.interface_data.t.shape[0]):

        #     if self.interface_data.cF[k]:
        #         colorx='r'
        #         colory='b'
        #         label = 'pusher in contact'
        #         in_contact = True
        #     else:
        #         colorx='orange'
        #         colory='purple'
        #         label = 'pusher not in contact'
        #         in_contact = False

        #     plt_configx = dict(markersize=0.5, color=colorx)
        #     plt_configy = dict(markersize=0.5, color=colory)
        #     if in_contact and is_first_r:
        #         plt_configx['label'] = 'x-pos in contact'
        #         plt_configy['label'] = 'y-pos in contact'
        #         is_first_r = False
        #     if (not in_contact) and is_first_b:
        #         plt_configx['label'] = 'x-pos not in contact'
        #         plt_configy['label'] = 'y-pos not in contact'
        #         is_first_b = False

        #     ax[0, 1].plot(self.interface_data.t[k], self.interface_data.xPW[k], 'o', **plt_configx)
        #     ax[0, 1].plot(self.interface_data.t[k], self.interface_data.yPW[k], 'o', **plt_configy)

        # Apply formatting to all figures
        for a in ax.flatten():
            a.grid()
            a.legend()

        self.ax_panel = ax

        print("Completed time evolution view plots")

    def show(self):
        plt.show()


# Load interface data using above class
interface_data_filename = os.path.join(
    os.getcwd(),
    'interface_data',
    # 'interface_data_wall.dat',
    # 'interface_data_tunnel.dat',
    'interface_data_tunnel_use.dat',
)
interface_data = InterfaceData(interface_data_filename)
interface_data.downsample_environment(150, random_seed=10)

# Plot data
dp = DataPlotter(interface_data)
dp.plot_birdseye()
dp.plot_panel()
dp.show()
