UROP WHAT I DID                                                                                                      
Two GitHub repos coz I’m dumb and didn’t really use GitHub …

repo 1:
UROP/script:
plane_estimation.py: White board EKF in python (yeah I tried to write it myself for a long time and didn’t work but Stefan did it in a night I cri.  Have a lot more to learn)

trajectory_3d.py: generate trajectory for input string, specific for font CNC in the font folder. Will output a trajectory txt. Need to pip install fonttools, ttfquery, numpy and matplotlib. Will crash if only enter a char…. hmmmm I should’ve fixed that….. parameters can be changed: vmax, amax, vmax_z, amax_z, text_size, z_distance, and point_spacing. vmax, amax: parameters in x and y axis in board frame, m/880/s. vmax_z, amax_z: parameters in z axis in board frame, m/880/s. text_size: size of text in meter. z_distance: distance of drone when not writing, in meter. point_spacing: the contour point I got from the font api has range 880 (-190 to 690 I remember?), so it is the spacing in unit 1/880/m. 

string_path.py: Trajectory of a string…. in 2d, dependencies same as above,

char_path.py: Trajectory of a char… in 2d, dependencies same as above, output the path of char in UROP/letters with ASCII code as file name. Mainly using graph theory but with a bit of a twig. 


UROP/src:
white_board_estimator.h&.cpp: White board EKF in cpp, currently generating random measurement per loop, can be modified to taking measurements

trajectory_publisher.cpp: publish the trajectory in Dimos’ message type

trajectory_listener.cpp: subscribe to the trajectory in Dimos’ message type


Thanks so much for letting me work with you guys. It has been a really fun experience! I learned a lot and I really enjoyed it. Stefan is a great supervisor and I wish I’m as smart as him xD. If you need help with the thing I did(well I didn’t put much comment in the code…. my fault) or anything else feel free to contact me via facebook/slack/email (qy916@ic.ac.uk)
