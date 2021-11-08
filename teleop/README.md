# teleop

Current pipeline.
1. Draw a new environment, i.e. obstacles, and start/goal positions using `draw_new_environment.py`
   - left button adds obstacles to scene, middle button adds start position, and right button adds goal position
   - when complete, press ESC, the script will save a data file containing the environment specification
1. User can then provide a demonstration via joystick by running `push_box_interface.py`
   - make sure to add a path to environment description (i.e. a file produced in previous script)
   - use joystick to provide input
   - use ESC to finish the demonstration
   - a data file containing the demonstration data is saved
