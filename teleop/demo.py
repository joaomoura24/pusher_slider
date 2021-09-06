import numpy
import pygame
import pygame_teleop  # https://github.com/cmower/pygame_teleop
from scipy.spatial.transform import Rotation
from scipy.integrate import dblquad
from scipy.linalg import block_diag

# Global constants
mu_g = 0.5  # coefficient of friction between the object and the ground
m = 1.0  # mass of object in kg
g = 9.807  # gravitational acceleration
box_width = 0.1  # width of box - along x axis
box_height = 0.05  # height of box - along y axis

# Compute constants
A = box_width*box_height
fmax = mu_g*m*g
mmax = (mu_g*m*g/A)*dblquad(lambda y, x: numpy.sqrt(x**2+y**2), -0.5*box_width, 0.5*box_width, -0.5*box_height, 0.5*box_height)[0]
L = numpy.diag([fmax, fmax, mmax], dtype=float)

# Setup system dynamics
side2vec = dict(
    right = (numpy.array([-1, 0], dtype=float), numpy.array([0, -1], dtype=float)),
    top = (numpy.array([0, -1], dtype=float), numpy.array([1, 0], dtype=float)),
    left = (numpy.array([1, 0], dtype=float), numpy.array([0, 1], dtype=float)),
    bottom = (numpy.array([0, 1], dtype=float), numpy.array([-1, 1], dtype=float)),

)

def f(x, u, r, side):

    # Extract state elements
    px = x[0]  # x position of object in world frame
    py = x[1]  # y position of object in world frame
    th = x[2]  # heading of object in world frame
    ph = x[3]  # angle of slider relative to object frame

    # Extract pusher elements
    xc = r[0]  # x position of slider in object frame
    yc = r[1]  # y position of slider in object frame

    # Compute dynamics
    R = Rotation.from_euler('z', th).as_matrix()  # 3-by-3
    J = numpy.array([[1, 0, -yc],
                     [0, 1,  xc]], dtype=float)

    n, np = side2vec[side]
    N = J.T @ n  # 3-by-2 * 2-by-1 -> 3-by-1
    T = J.T @ np  # 3-by-2 * 2-by-1 -> 3-by-1
    B = numpy.concatenate((N, T), axis=1)  # 3-by-2

    return block_diag(R@L@B, 1.0) @ u  # 4-by-3 * 3-by-1 = 4-by-1


# Main program
def main():

    # Setup
    config = {
        'caption': 'Drawing example',
        'width': 500,
        'height': 500,
        'background_color': 'darkslateblue',
        'windows':{
            'robotenv': {
                'origin': (10, 10),
                'width': 480,
                'height': 480,
                'background_color': 'white',
                'type': 'RobotEnvironment',
                'robotenv_width': 1.0,
                'robotenv_height': 1.0,
                'robotenv_origin_location': 'lower_left',
                'show_origin': True,
                'robots': {
                    'robot1': {
                        'show_path': False,
                        'robot_radius': 0.025,
                        'robot_color': 'black'
                    }
                }

            }
        }
    }
    screen = Screen(config)
    clock = pygame.time.Clock()
    draw_colors = ['red', 'green', 'blue']
    draw_color = None
    hz = 80
    draw = False
    running = True

    # Main loop
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    draw = True
                    draw_color = draw_colors[event.button-1]

                if event.type == pygame.MOUSEBUTTONUP:
                    draw = False

            pos = screen.windows['robotenv'].get_mouse_position()

            screen.reset()
            screen.windows['robotenv'].robots['robot1'].draw(pos)
            if draw:
                screen.windows['robotenv'].static_circle(
                    draw_color,
                    screen.windows['robotenv'].convert_position(pos),
                    screen.windows['robotenv'].convert_scalar(0.02),
                )
            screen.final()
            clock.tick(hz)
    except KeyboardInterrupt:
        pass

    pygame.quit()
    print("Goodbye")


if __name__ == '__main__':
    main()
