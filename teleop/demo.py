import numpy
import pygame
import pygame_teleop  # https://github.com/cmower/pygame_teleop
from numpy import sign
from scipy.spatial.transform import Rotation
from scipy.integrate import dblquad
from scipy.linalg import block_diag
from math import sin, cos
from pygame_teleop.screen import Screen
from pygame_teleop.joystick import Joystick

"""Needs a joystick with at least 3 axes: hori/vert/rot. Otherwise,
otherwise the keyboard is used."""

numpy.set_printoptions(precision=2, suppress=True)

# Global constants
hz = 80
dt = 1.0/float(hz)
pi = numpy.pi
mu_g = 0.5  # coefficient of friction between the object and the ground
m = 1.0  # mass of object in kg
g = 9.807  # gravitational acceleration
slider_width = 0.3  # width of slider - along x axis
slider_height = 0.2  # height of slider - along y axis

# Compute constants
A = slider_width*slider_height  # area of slider
fmax = mu_g*m*g
mmax = (mu_g*m*g/A)*dblquad(lambda y, x: numpy.sqrt(x**2+y**2), -0.5*slider_width, 0.5*slider_width, -0.5*slider_height, 0.5*slider_height)[0]
L = numpy.diag([1/fmax**2, 1/fmax**2, 1/mmax**2])

# Setup system dynamics
side2vec = dict(
    right = (numpy.array([-1, 0], dtype=float), numpy.array([0, -1], dtype=float)),
    top = (numpy.array([0, -1], dtype=float), numpy.array([1, 0], dtype=float)),
    left = (numpy.array([1, 0], dtype=float), numpy.array([0, 1], dtype=float)),
    bottom = (numpy.array([0, 1], dtype=float), numpy.array([-1, 1], dtype=float)),
)

def xdot(x, u, r, side):

    # Extract state elements
    px = x[0]  # x position of object in world frame
    py = x[1]  # y position of object in world frame
    th = x[2]  # heading of object in world frame
    ph = x[3]  # angle of pusher relative to object frame

    # Extract pusher elements
    xc = r[0]  # x position of pusher in object frame
    yc = r[1]  # y position of pusher in object frame

    # Compute dynamics
    R = Rotation.from_euler('z', th).as_matrix()  # 3-by-3
    J = numpy.array([[1, 0, -yc],
                     [0, 1,  xc]], dtype=float)

    n, np = side2vec[side]
    N = J.T @ n  # 3-by-2 * 2-by-1 -> 3-by-1
    T = J.T @ np  # 3-by-2 * 2-by-1 -> 3-by-1

    B = numpy.zeros((3, 2))
    B[:, 0] = N
    B[:, 1] = T

    return block_diag(R@L@B, 1.0) @ u  # 4-by-3 * 3-by-1 = 4-by-1


class Point:

    def __init__(self, x, y, slider, mass=1.0):
        self.pos = numpy.asarray([x, y], dtype=float)
        self.vel = numpy.zeros(2)
        self.acc = None
        self.slider = slider
        p = self.pos_in_slider_frame()
        self.angle = numpy.arctan2(p[1], p[0])
        self.angle_vel = None
        self.m = mass

    def update(self, pos):
        old_vel = self.vel.copy()
        vel = (pos - self.pos)/dt
        self.pos = pos
        self.vel = vel
        self.acc = (self.vel - old_vel)/dt
        self.force = self.acc * self.m

        old_angle = self.angle
        p = self.pos_in_slider_frame()
        new_angle = numpy.arctan2(p[1], p[0])
        if sign(old_angle) == 1 and sign(new_angle) == -1:
            new_angle += 2*pi
        elif sign(old_angle) == -1 and sign(new_angle) == 1:
            old_angle += 2*pi

        self.angle = new_angle
        self.angle_vel = (new_angle - old_angle)/dt

    def pos_in_world_frame(self):
        return self.pos

    def pos_in_slider_frame(self):
        R = Rotation.from_euler('z', self.slider.heading, degrees=False).as_matrix()[:2,:2]
        r = R.T@(self.pos - self.slider.pos)
        return r

    def force_in_slider_frame(self):
        R = Rotation.from_euler('z', self.slider.heading, degrees=False).as_matrix()[:2,:2]
        return R.T@self.force


class Slider:

    contact_tol = 0.01

    def __init__(self, x, y, heading, width, height):
        self.pos = numpy.asarray([x, y], dtype=float)
        self.heading = heading
        self.width = width
        self.height = height

        # left/top/right/bottom in slider frame
        self.left = -0.5*width
        self.top = 0.5*height
        self.right = 0.5*width
        self.bottom = -0.5*height

    def update(self, pos, heading):
        self.pos = pos
        self.heading = heading

    def near_point(self, point):
        p = point.pos_in_slider_frame()
        x = numpy.clip(p[0], self.left, self.right)
        y = numpy.clip(p[1], self.bottom, self.top)
        dl = abs(x-self.left)
        dr = abs(x-self.right)
        dt = abs(y-self.top)
        db = abs(y-self.bottom)
        m = min(dl, dr, dt, db)
        if m == dt:
            pos_in_slider_frame = numpy.array([x, self.top])
            side = 'top'
        elif m == db:
            pos_in_slider_frame = numpy.array([x, self.bottom])
            side = 'bottom'
        elif m == dl:
            pos_in_slider_frame = numpy.array([self.left, y])
            side = 'left'
        else:
            pos_in_slider_frame = numpy.array([self.right, y])
            side = 'right'
        R = Rotation.from_euler('z', self.heading, degrees=False).as_matrix()[:2,:2]
        in_contact = numpy.linalg.norm(pos_in_slider_frame - p) <= self.contact_tol
        return self.pos + R@pos_in_slider_frame, side, in_contact


# Main program
def main():

    # Setup
    config = {
        'caption': 'Pushing teleop',
        'width': 1000,
        'height': 1000,
        'background_color': 'darkslateblue',
        'windows':{
            'robotenv': {
                'origin': (10, 10),
                'width': 1000-20,
                'height': 1000-20,
                'background_color': 'white',
                'type': 'RobotEnvironment',
                'robotenv_width': 1.0,
                'robotenv_height': 1.0,
                'robotenv_origin_location': 'lower_left',
                'show_origin': True,
                'robots': {
                    'robot1': {
                        'show_path': False,
                        'robot_radius': 0.005,
                        'robot_color': 'black'
                    },
                    'pusher': {
                        'show_path': False,
                        'robot_radius': 0.01,
                        'robot_color': 'blue'
                    },
                }

            }
        }
    }
    screen = Screen(config)
    clock = pygame.time.Clock()
    draw_colors = ['red', 'green', 'blue']
    draw_color = None
    slider = Slider(0.3, 0.2, 0, slider_width, slider_height)
    pusher = numpy.array([0.25*slider_width, slider.bottom])
    pusher_angle = numpy.arctan2(pusher[1], pusher[0])
    side = 'bottom'
    fmax = 1.5
    max_avel = numpy.deg2rad(10)
    try:
        joy = Joystick()
        use_joy = True
    except pygame.error:
        use_joy = False
        j = [0.0, 0.0, 0.0, 0.0]
    running = True

    # Main loop
    try:
        while running:

            # Update user input j
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        j[3] = -1.0
                    elif event.key == pygame.K_RIGHT:
                        j[3] = 1.0
                    elif event.key == pygame.K_UP:
                        j[1] = -1.0
                    elif event.key == pygame.K_z:
                        j[0] = -1.0
                    elif event.key == pygame.K_x:
                        j[0] = 1.0
                elif event.type == pygame.KEYUP:
                    if (event.key == pygame.K_LEFT) or (event.key == pygame.K_RIGHT):
                        j[3] = 0.0
                    elif event.key == pygame.K_UP:
                        j[1] = 0.0
                    elif (event.key == pygame.K_z) or (event.key == pygame.K_x):
                        j[0] = 0.0

            if use_joy:
                joy.reset()
                j = joy.get_axes()

            # Update pusher/slider state
            old_pusher = pusher.copy()
            pusher[0] += j[3]*max_avel*dt
            pusher[0] = numpy.clip(pusher[0], -0.5*slider_width, 0.5*slider_width)
            new_pusher_angle = numpy.arctan2(pusher[1], pusher[0])
            angle_vel = (new_pusher_angle - pusher_angle)/dt
            pusher_angle = new_pusher_angle
            x = numpy.array([slider.pos[0], slider.pos[1], slider.heading, pusher_angle])
            fnc = -fmax*numpy.clip(j[1], -1, 0)
            ftc = numpy.clip(-fmax*j[0], -mu_g*fnc, mu_g*fnc)
            u = numpy.array([fnc, ftc, angle_vel])
            x += dt*xdot(x, u, pusher, side)
            slider.update(x[:2], x[2])
            R = Rotation.from_euler('z', slider.heading, degrees=False).as_matrix()[:2,:2]
            pusher_in_world = R@pusher + slider.pos

            # Draw
            screen.reset()
            screen.windows['robotenv'].draw_box('red', slider.width, slider.height, slider.pos, slider.heading)
            screen.windows['robotenv'].robots['pusher'].draw(pusher_in_world)
            screen.final()
            clock.tick_busy_loop(hz)


    except KeyboardInterrupt:
        pass

    pygame.quit()
    print("Goodbye")


if __name__ == '__main__':
    main()
