import os
import pickle
import datetime
import pygame
from copy import deepcopy
from pygame_teleop.screen import Screen

def main():

    obs_radius = 0.05

    # Setup
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
                        'robot_radius': obs_radius,
                        'show_path': False,
                        'robot_color': 'red',
                    },
                },
            },
        },
    }
    screen = Screen(config)
    clock = pygame.time.Clock()
    draw_colors = ['red', 'green', 'blue']
    draw_color = 'black'
    hz = 80
    draw = False
    running = True
    obs_data = []
    start_pos = None
    end_pos = None

    # Main loop
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        draw = True
                        draw_color = 'black'
                    elif event.button == 2:
                        draw = True
                        draw_color = 'green'
                    elif event.button == 3:
                        draw = True
                        draw_color = 'red'

                if event.type == pygame.MOUSEBUTTONUP:
                    draw = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            pos = screen.windows['robotenv'].get_mouse_position()

            if draw and (draw_color == 'black'):
                d = (deepcopy(pos[0]), deepcopy(pos[1]), obs_radius)
                obs_data.append(d)
            if draw and (draw_color == 'green'):
                start_pos = deepcopy(pos)
            if draw and (draw_color == 'red'):
                end_pos = deepcopy(pos)

            screen.reset()
            screen.windows['robotenv'].robots['pusher'].draw(pos)
            if draw:
                screen.windows['robotenv'].static_circle(
                    draw_color,
                    screen.windows['robotenv'].convert_position(pos),
                    screen.windows['robotenv'].convert_scalar(obs_radius),
                )
            screen.final()
            clock.tick(hz)
    except KeyboardInterrupt:
        pass

    pygame.quit()


    if len(obs_data) == 0:
        print("WARN: no obstacle data given!")
    if start_pos is None:
        print("WARN: start position not given!")
    if end_pos is None:
        print("WARN: end position not given!")


    timestamp = datetime.datetime.now()
    filename = 'env_data' + timestamp.strftime('_%Y-%m-%d-%H-%M-%S') + '.dat'
    filename_full = os.path.join(os.getcwd(), 'environment_descriptions', filename)
    data = {'obs': obs_data, 'start': start_pos, 'goal': end_pos}
    with open(filename_full, 'wb') as f:
        pickle.dump(data, f)
    print("Saved", filename, 'in environment_descriptions/')


    print("Goodbye")


if __name__ == '__main__':
    main()
