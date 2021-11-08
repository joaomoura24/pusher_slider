import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

cx, cy = 0.75, 0.6
radius = 0.1
rx, ry = 0.48, 0.43
rw, rh = 0.2, 0.1

def in_collision():
    # Check if circle and rectangle are in collision: http://www.jeffreythompson.org/collision-detection/circle-rect.php
    # cx - x-position of circle
    # cy - y-position of circle
    # r - radius of circle
    # rx - x-position of the bottom left hand corner of the rectangle
    # ry - y-position of the bottom left hand corner of the rectangle
    # rw - width of rectangle
    # rh - height of rectangle

    # Temporary variables to set edges for testing
    testX = float(cx)
    testY = float(cy)

    # Which edge is closest?
    if (cx < rx):
        testX = rx      # test left edge
    elif (cx > rx+rw):
        testX = rx+rw   # right edge
    if (cy < ry):
        testY = ry      # top edge
    elif (cy > ry+rh):
        testY = ry+rh   # bottom edge

    # Get distance from closest edges
    distX = cx-testX
    distY = cy-testY
    distance = numpy.sqrt((distX*distX) + (distY*distY))

    # If the distance is less than the radius, collision!
    if (distance <= radius):
        return True
    return False



fig, ax = plt.subplots(tight_layout=True)
ax.set_aspect('equal')

ax.add_patch(plt.Circle((cx, cy), radius, color='b'))

if in_collision():
    c = 'red'
else:
    c = 'green'
ax.add_patch(Rectangle((rx, ry), rw, rh, color=c))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.grid()


plt.show()
