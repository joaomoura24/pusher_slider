# teleop

## Data

For each trial, two data files are saved.
The filenames for the two files have the following structure `data_STAMP.csv` and `data_STAMP.config` where `STAMP` is a unique time stamp, these are time-series data and the configuration respectively.
See the demo scripts for an explanation of the meaning of each parameter in the configuration data.

Columns of `.csv` file:
- `t`: time stamp (secs)
- `x0`: x-position of object in world frame
- `x1`: y-position of object in world frame
- `x2`: heading of object in world frame
- `x3`: angle of pusher relative to slider frame
- `u0`: applied normal force magnitude at contact point resolved in the body frame
- `u1`: applied tangential force magnitude at contact point resolved in the body frame
- `u2`: commanded angular velocities resolved in the body frame
- `side`: side of slider that the pusher interacts with
