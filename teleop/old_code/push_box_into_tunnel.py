import os
import pygame
import time
import numpy
import pickle
import datetime
import pygame_teleop
from math import sin, cos, sqrt, tan
from pygame_teleop.screen import Screen
from pygame_teleop.joystick import Joystick
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
from sklearn.preprocessing import normalize
from scipy.integrate import dblquad

# -------------------------------------------------------------------------------------------
#
# Helper functions
# -------------------------------------------------------------------------------------------

def print_debug(msg):
    print(pygame.time.get_ticks(), msg)

def nearest_point_on_perimeter_of_rectangle(w, h, x, y):
    """Given a point in space, compute the point on perimeter of a rectangle with smallest distance. Returns position and side id."""
    # https://stackoverflow.com/a/20453634
    # https://gist.github.com/dwtkns/d5b9b60285b8b0067c53
    # Note, I tried to implement this using dwtkns's gist, but this seemed to be wrong
    # I must have solved this before, I found the implementation below in an old backup file

    left = -0.5*w
    top = 0.5*h
    right = 0.5*w
    bottom = -0.5*h

    x = numpy.clip(x, left, right)
    y = numpy.clip(y, bottom, top)

    dl = abs(x-left)
    dr = abs(x-right)
    dt = abs(y-top)
    db = abs(y-bottom)
    m = min(dl, dr, dt, db)
    if m == dt:
        p = numpy.array([x, top])
        side = 2
    elif m == db:
        p = numpy.array([x, bottom])
        side = 4
    elif m == dl:
        p = numpy.array([left, y])
        side = 3
    else:
        p = numpy.array([right, y])
        side = 1
    return p, side

# -------------
# Test: nearest_point_on_perimeter_of_rectangle
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.animation import FuncAnimation
# w = 0.3
# h = 0.3
# radius = 1.1*max(w, h)
# fig, ax = plt.subplots(tight_layout=True)
# ax.set_aspect('equal')
# ax.set_xlim(-2*max(h, w), 2*max(h, w))
# ax.set_ylim(-2*max(h, w), 2*max(h, w))
# ax.add_patch(plt.Circle((0, 0), radius, color='g'))
# ax.add_patch(Rectangle((-0.5*w, -0.5*h), w, h, 0))
# line, = ax.plot([], [], '-or', lw=2)
# def init():
#     line.set_data([], [])
#     return line,
# def animate(th):
#     x = radius*cos(th)
#     y = radius*sin(th)
#     p, side = nearest_point_on_perimeter_of_rectangle(w, h, x, y)
#     line.set_data([p[0], x], [p[1], y])
#     return line,
# anim = FuncAnimation(fig, animate, init_func=init, frames=numpy.linspace(0, 2*numpy.pi, 1000), interval=20, blit=True)
# plt.show()
# quit()
# end of test
# -------------

def point_on_rectangle_given_angle(phi, w, h):
    # https://math.stackexchange.com/a/1704732
    a = 0.5*w
    b = 0.5*h
    if abs(tan(phi)) <= b/a:
        r = a/abs(cos(phi))
    else:
        r = b/abs(sin(phi))
    x = r*cos(phi)
    y = r*sin(phi)
    return numpy.array([x, y])

# -------------
# Test: point_on_rectangle_given_angle
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.animation import FuncAnimation
# w = 1
# h = 1.5
# fig, ax = plt.subplots(tight_layout=True)
# ax.set_aspect('equal')
# ax.set_xlim(-2*max(h, w), 2*max(h, w))
# ax.set_ylim(-2*max(h, w), 2*max(h, w))
# ax.add_patch(Rectangle((-0.5*w, -0.5*h), w, h, 0))
# line, = ax.plot([], [], '-or', lw=2)
# def init():
#     line.set_data([], [])
#     return line,
# def animate(th):
#     p = point_on_rectangle_given_angle(th, w, h)
#     line.set_data([0, p[0]], [0, p[1]])
#     return line,
# anim = FuncAnimation(fig, animate, init_func=init, frames=numpy.linspace(0, 2*numpy.pi, 1000), interval=20, blit=True)
# plt.show()
# quit()
# end of test
# -------------

# -------------------------------------------------------------------------------------------
#
# Helpful classes
# -------------------------------------------------------------------------------------------

class Data:

    def __init__(self):
        self.t = []  # time, real time using python time.time() method
        self.x = []  # state trajectory
        self.cf = []  # contact flag: True means in contact, False otherwise

    def save(self):

        # Get data
        t = numpy.array(self.t)-self.t[0]
        x = numpy.array(self.x).T
        cf = numpy.array(self.cf, dtype=bool)

        # Save
        timestamp = datetime.datetime.now()
        filename = 'data_' + os.path.basename(__file__) + timestamp.strftime('_%Y-%m-%d-%H-%M-%S') + '.dat'
        with open(filename, 'wb') as f:
            pickle.dump({'x': x, 't': t, 'cf': self.cf}, f)
        print("Saved", filename)

# -------------------------------------------------------------------------------------------
#
# Parameters and variables
# -------------------------------------------------------------------------------------------

# User specified
body_width = 0.5  # width of body - along x axis
body_height = 0.5  # height of body - along y axis
mu_g = 0.5  # coefficient of friction between the body and the surface
m = 1.0  # mass of slider in kg
g = 9.807  # gravitational acceleration
hz = 50
max_pusher_speed = 0.3  # maximum speed of pusher
max_pusher_normal_force = 7  # maximum force normal to surface
max_pusher_angular_velocity = numpy.deg2rad(90)  # maximum angular velocity of pusher in body frame
pi = numpy.pi
xB0 = 1.8  # initial x-position of body in world
yB0 = -0.7  # initial y-position of body in world
thetaB0 = -pi/2.0  # initial heading of body in world
phiP0 = numpy.deg2rad(90)  # initial angle of pusher in body frame
pusher_body_dist_tol = 0.01  # tolerance distance between pusher and body, when distance between pusher/body less than this, the pusher snaps to the body

goalX = 0  # x-position of goal of body
goalY = 0.5 # y-position of goal of body
goalTheta = -pi/2.0  # theta heading for goal of body

goal_position = numpy.array([goalX, goalY])  # goal position for body
body_radius = 0.354  # radius around body (for collision avoidance and making sure tunnel is wide enough)
Nobs_inset = 20  # number of obstacles in a line set
obs_radius = 0.05  # radius of an obstacle
tunnel_depth = 0.9  # depth of tunnel
tunnel_pad = obs_radius + 0.01#0.02  # padding between obs radius and tunnel
wall_length = 1.5  # length of a wall segment
tunnel_yshift = -0.8  # shift wall in y-direction
tunnel_center_shift = numpy.array([0, tunnel_yshift]) # np.array([0, -0.2]) #np.array([0, -0.3])
tunnel_center = goal_position + tunnel_center_shift  # center of tunnel object
xobs1 = tunnel_center[0] + tunnel_pad + body_radius*numpy.ones(Nobs_inset)
yobs1 = tunnel_center[1] + numpy.linspace(0, tunnel_depth, Nobs_inset)
xobs2 = tunnel_center[0] + numpy.linspace(tunnel_pad + body_radius, wall_length, Nobs_inset)
yobs2 = tunnel_center[1] + numpy.zeros(Nobs_inset)
xobs3 = tunnel_center[0] - tunnel_pad - body_radius*numpy.ones(Nobs_inset)
yobs3 = tunnel_center[1] + numpy.linspace(0, tunnel_depth, Nobs_inset)
xobs4 = tunnel_center[0] + numpy.linspace(-wall_length, -tunnel_pad - body_radius, Nobs_inset)
yobs4 = tunnel_center[1] + numpy.zeros(Nobs_inset)

# Other
obs1 = numpy.stack((xobs1, yobs1))
obs2 = numpy.stack((xobs2, yobs2))
obs3 = numpy.stack((xobs3, yobs3))
obs4 = numpy.stack((xobs4, yobs4))
obs_data = [obs1, obs2, obs3, obs4]
A = body_width*body_height  # area of slider
fmax = mu_g*m*g  # semi-principle axes for ellipsoid
mmax = (mu_g*m*g/A)*dblquad(lambda y, x: numpy.sqrt(x**2+y**2), -0.5*body_width, 0.5*body_width, -0.5*body_height, 0.5*body_height)[0]   # semi-principle axes for ellipsoid
L = numpy.diag([1/fmax**2, 1/fmax**2, 1/mmax**2])  # positive definite matrix defining limit surface
R90 = Rotation.from_euler('z', 90, degrees=True).as_matrix()[:2, :2]  # rotation matrix through 90 degrees
dt = 1.0/float(hz)  # time step
phiP0 = numpy.fmod(phiP0, 2*pi)  # ensure phiP0 in range [0, 2pi)
pP0 = point_on_rectangle_given_angle(phiP0, body_width, body_height)  # point of pusher
x = numpy.array([xB0, yB0, thetaB0, phiP0, pP0[0], pP0[1]])  # system state
data = Data()

# -------------------------------------------------------------------------------------------
#
# Setup pygame
# -------------------------------------------------------------------------------------------

w0 = 20
w1 = 900

config = {
    'caption': 'Pushing box change of contact',
    'width': 2*w0+w1,
    'height': 2*w0+w1,
    'background_color': 'darkslateblue',
    'windows': {
        'robotenv': {
            'origin': (w0, w0),
            'width': w1,
            'height': w1,
            'show_origin': False,
            'robotenv_origin_location': 'center',
            'background_color': 'white',
            'type': 'RobotEnvironment',
            'robotenv_width': 5.0,
            'robotenv_height': 5.0,
            'robots': {
                'pusher': {
                    'robot_radius': 0.025,
                    'show_path': False,
                    'robot_color': 'blue',
                },
            },
        },
    },
}

screen = Screen(config)
for obs in obs_data:
    for i in range(Nobs_inset):
        screen.windows['robotenv'].static_circle(
            'black',
            screen.windows['robotenv'].convert_position(obs[:, i]),
            screen.windows['robotenv'].convert_scalar(obs_radius)
        )

dr_goal_ax = numpy.array([cos(goalTheta), sin(goalTheta)])
goal_xaxis = goal_position + 0.75*body_radius*dr_goal_ax
goal_yaxis = goal_position + 0.75*body_radius*R90@dr_goal_ax

goal_pos_img = screen.windows['robotenv'].convert_position(goal_position)
goal_xaxis_img = screen.windows['robotenv'].convert_position(goal_xaxis)
goal_yaxis_img = screen.windows['robotenv'].convert_position(goal_yaxis)

screen.windows['robotenv'].static_line('red', goal_pos_img, goal_xaxis_img, width=2)
screen.windows['robotenv'].static_line('green', goal_pos_img, goal_yaxis_img, width=2)

clock = pygame.time.Clock()
running = True

# -------------------------------------------------------------------------------------------
#
# Joystick
# -------------------------------------------------------------------------------------------

class Joy:

    def __init__(self):
        self._joy = Joystick()

    def h_for_pusher_model(self, *args):
        """Forward joy maps pushes in direction of normal, left/right pushes in tangential direction."""
        self._joy.reset()
        raw = self._joy.get_axes()
        fn = -max_pusher_normal_force*numpy.clip(raw[1], -1, 0)  # force on pusher normal to surface
        ft = numpy.clip(-max_pusher_normal_force*raw[0], -mu_g*fn, mu_g*fn)  # force on pusher tangent to surface
        phidot = max_pusher_angular_velocity*raw[-1]
        return numpy.array([fn, ft, phidot])

    # def h_for_point_model(self, *args):
    #     """Aligned with local body frame"""
    #     self._joy.reset()
    #     return numpy.array([1, -1])*max_pusher_speed*numpy.array(Joystick.isometric(self._joy.get_axes()[:2]))

    def h_for_point_model(self, *args):
        """Aligned with world frame."""
        self._joy.reset()
        x = args[0]
        h = numpy.array([1, -1])*max_pusher_speed*numpy.array(Joystick.isometric(self._joy.get_axes()[:2]))  # vel of pusher in world
        R = Rotation.from_euler('z', -x[2]).as_matrix()[:2, :2]  # rotation matrix for body
        return R@h

joy = Joy()

# -------------------------------------------------------------------------------------------
#
# Model of dynamics
# -------------------------------------------------------------------------------------------

def dyn_pusher(x, u, side):

    # Extract state elements
    xB = x[0]  # x-position of body in world
    yB = x[1]  # y-position of body in world
    thetaB = x[2]  # heading of body in world
    phiP = x[3]  # angle of pusher in body frame
    xP = x[4]  # x position of pusher in body frame
    yP = x[5]  # y position of pusher in body frame

    # Extract (required) control elements
    phidot = u[2]  # commanded angular velocity of pusher in body frame

    # Compute dynamics using Hogan model
    R = Rotation.from_euler('z', thetaB).as_matrix()  # rotation matrix for body pose in world frame
    J = numpy.array([[1.0, 0.0, -yP],  # Jacobian matrix associated with the contact point in body frame
                     [0.0, 1.0,  xP]])
    if side == 1:
        # right side
        n = numpy.array([-1, 0])  # normal vector to surface in body frame
    elif side == 2:
        # top side
        n = numpy.array([0, -1])
    elif side == 3:
        # left side
        n = numpy.array([1, 0])
    elif side == 4:
        # bottom side
        n = numpy.array([0, 1])
    else:
        raise ValueError(f'did not recognize given side id (was given {side})')

    B = numpy.zeros((3, 2))  # temporary matrix
    B[:, 0] = J.T@n  # matrix of object normal vectors at contact points resolved in body frame
    B[:, 1] = J.T@R90@n  # matrix of object tangent vectors at contact points resolved in body frame

    xdot = numpy.zeros(6)
    xdot[:4] = block_diag(R@L@B, 1.0) @ u  # Model of dynamics by Hogan

    # Compute dynamics of pusher point mass
    r = sqrt(xP**2 + yP**2)  # displacement of pusher from body frame origin
    if side == 1 or side == 3:
        vxP = 0.0  # velocity of pusher in body x-axis
        vyP = phidot*xP/(1-yP*sin(phiP)/r)  # velocity of pusher in body y-axis
    else:
        # side is 2 or 4 (above if-statement confirms this)
        vxP = phidot*yP/(xP*cos(phiP)/r - 1)  # velocity of pusher in body x-axis
        vyP = 0.0  # velocity of pusher in body y-axis

    xdot[4] = vxP
    xdot[5] = vyP

    return xdot


def dyn_point(x, u):

    # Extract state elements
    xB = x[0]  # x-position of body in world
    yB = x[1]  # y-position of body in world
    thetaB = x[2]  # heading of body in world
    phiP = x[3]  # angle of pusher in body frame
    xP = x[4]  # x position of pusher in body frame
    yP = x[5]  # y position of pusher in body frame

    # Extract control elements
    vxP = u[0]  # velocity of pusher in body x-axis
    vyP = u[1]  # velocity of pusher in body y-axis

    # Compute dynamics
    xdot = numpy.zeros(6)
    r = sqrt(xP**2 + yP**2)  # displacement of pusher from body frame origin
    rdot = xP*vxP + yP*vyP/r  # rate of change of pusher displacement from body frame origin
    phidot = (rdot*cos(phiP) - vxP)/yP  # angular velocity of pusher in body frame
    xdot[3] = phidot
    xdot[4] = vxP
    xdot[5] = vyP

    return xdot

DYN_POINT_MODEL = 0
DYN_PUSHER_MODEL = 1

dynamics_model = DYN_PUSHER_MODEL

# -------------------------------------------------------------------------------------------
#
# Main loop
# -------------------------------------------------------------------------------------------

# Append initial state to data
data.t.append(time.time())
data.x.append(x.tolist())
data.cf.append(True)  # always starts in contact

# Main loop
while running:

    # For profiling main loop
    # t0 = time.time()

    # Handle pygame events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                dynamics_model = DYN_POINT_MODEL

            if event.key == pygame.K_ESCAPE:
                running = False

        elif event.type == pygame.KEYUP:

            if event.key == pygame.K_SPACE:
                dynamics_model = DYN_PUSHER_MODEL

    # Compute nearest point on body to pusher and the distance between them
    near_point, side = nearest_point_on_perimeter_of_rectangle(body_width, body_height, x[4], x[5])
    near_dist = numpy.linalg.norm(x[4:] - near_point)

    # Update state
    if dynamics_model == DYN_POINT_MODEL:
        h = joy.h_for_point_model(x, side)
        xdot = dyn_point(x, h)
        in_contact = False
    elif (dynamics_model == DYN_PUSHER_MODEL) and near_dist > pusher_body_dist_tol:
        # space released, now move robot to body side (nearest point)
        vP = 0.8*max_pusher_speed*normalize((near_point - x[4:]).reshape(1, -1)).flatten()
        xdot = dyn_point(x, vP)
        in_contact = False
    else:
        # ensure pusher is on surface of body and compute xdot
        # x[4:] = point_on_rectangle_given_angle(x[3], body_width, body_height)
        p = x[4:]
        x[3] = numpy.arctan2(p[1], p[0])
        x[4:] = point_on_rectangle_given_angle(x[3], body_width, body_height)
        h = joy.h_for_pusher_model(x, side)
        xdot = dyn_pusher(x, h, side)
        in_contact = True

    x += dt*xdot

    # Compute pusher position in world
    R = Rotation.from_euler('z', x[2]).as_matrix()[:2, :2]  # rotation matrix for body
    pW = R@x[4:] + x[:2]

    # Compute line for x-axis for body frame
    xaxis = numpy.zeros((2, 2))
    xaxis[:, 0] = x[:2]
    xaxis[:, 1] = R@(0.75*0.5*min(body_width, body_height)*numpy.array([1, 0])) + x[:2]

    # Compute line for y-axis for body frame
    yaxis = numpy.zeros((2, 2))
    yaxis[:, 0] = x[:2]
    yaxis[:, 1] = R@(0.75*0.5*min(body_width, body_height)*numpy.array([0, 1])) + x[:2]

    # Update visualization
    screen.reset()
    screen.windows['robotenv'].draw_box('brown', body_width, body_height, x[:2], x[2])
    screen.windows['robotenv'].draw_path('red', xaxis, width=4)
    screen.windows['robotenv'].draw_path('green', yaxis, width=4)
    screen.windows['robotenv'].robots['pusher'].draw(pW)
    screen.windows['robotenv'].draw_box('green', body_width, body_height, goal_position, rotation=goalTheta, alpha=100)
    screen.final()

    # Sleep
    # t1 = time.time()
    # print("%.2f"%(1/(t1-t0)))
    clock.tick_busy_loop(hz)

    # Update data
    data.t.append(time.time())
    data.x.append(x.tolist())
    data.cf.append(in_contact)

# -------------------------------------------------------------------------------------------
#
# Finish
# -------------------------------------------------------------------------------------------

data.save()
print("Goodbye")
