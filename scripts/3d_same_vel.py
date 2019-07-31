import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time


to_write = input("Enter a word: \n")
# to_write = "NaTuRaL wRiTiNg"
list_char = list(to_write)
width = 0
angle_threshold = 130/180*np.pi
point_spacing = 100

font_url = "fonts/cnc_v.ttf"
font = describe.openFont(font_url)
# v = 500
timestp = 0.0
space_width = 400
text_size = 1
z_distance = 0.2


amax = 50
vmax = 20
vmax_z = 10
amax_z = 50


def find_angle(point1,point2,point3):
	a = distance(point1,point2)
	b = distance(point2,point3)
	c = distance(point3,point1)
	angle = np.arccos((a**2+b**2-c**2)/(2*a*b))
	return angle

def distance(a,b):
	return np.linalg.norm(a-b)

def flatten_list(lst):
	return [item for sublist in lst for item in sublist]

# split the list based on given indices
def indexsplit(some_list, indices):
    indices = [i+1 for i in indices]
    indices.insert(0,0)
    my_list = []
    for start, end in zip(indices, indices[1:]):
        my_list.append(some_list[start:end])
    return my_list

# find the trajectory of one char
def find_char_trajectory(char,width):
	x = []
	y = []
	contact = []
	filename = "letters/{}.txt".format(ord(char))
	with open(filename, 'r') as f: 
		lines = f.readlines()
		for line in lines:
			x.append(int(line.split(' ')[0]))
			y.append(int(line.split(' ')[1]))
			contact.append(int(line.split(' ')[2]))
	x = [element+width for element in x]
	contact_array_char = np.array(contact)
	indices = np.where(contact_array_char==0)[0].tolist()
	x_list = indexsplit(x,indices)
	y_list = indexsplit(y,indices)
	return x_list,y_list,contact

# find the trajectory of the whole word
def find_whole_trajectory(list_char):
	x_all = []
	y_all = []
	contact_all = []
	global width
	print("calculating trajectory...")
	for char in list_char:
		if char.isspace():
			width += space_width
		else:
			x,y,contact = find_char_trajectory(char,width)
			glyph_name = glyphquery.glyphName(font,char)
			width += glyphquery.width(font,glyph_name)
			contact_all.append(contact)
			if any(isinstance(ls,list) for ls in x):
				for i in range(len(x)):
					x_all.append(x[i])
					y_all.append(y[i])
			else:
				x_all.append(x)
				y_all.append(y)
	return x_all,y_all,contact_all


def plot_colourline(x,y,c):
    c = cm.plasma([v/vmax for v in c])
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]],c=c[i])
    return

def find_key_distance_long(total_distance,vmax,amax):
	s1 = vmax**2/(2*amax)
	s2 = total_distance - s1
	t1 = vmax/amax
	t2 = (s2-s1)/vmax+t1
	return s1, s2,t1,t2

def find_key_distance_short(total_distance,amax):
	s = total_distance/2
	t = np.sqrt(total_distance/amax)
	return s,t

# ONLY 2D
def find_segment(path,angle_threshold,contact): # only feed in the x and y path ty
	segment_contact = []
	start_index = 0
	for i in range(1,len(contact)):
		if contact[i] == 0 and contact[i-1] == 1:
			segment_contact.append(path[start_index:i+1])
			start_index = i
		elif contact[i] == 1 and contact[i-1] == 0:
			segment_contact.append(path[start_index:i+1])
			start_index = i
		else:
			pass
	segment_contact.append(path[start_index:])
	segment_list = []
	for segment in segment_contact:
		angle_array = np.array([])
		if len(segment) > 3:
			for i in range(len(segment)-2):
				angle = find_angle(path[i],path[i+1],path[i+2])
				angle_array = np.append(angle_array,angle)
			# print(angle_array)
			segment_index = np.argwhere(angle_array<angle_threshold)
			segment_index = segment_index.T[0]
			n = len(segment_index)
			if n > 0:
				segment_list.append(segment[:segment_index[0]+2,:])
				for i in range(n-1):
					segment_list.append(path[segment_index[i]+1:segment_index[i+1]+2,:])
				segment_list.append(segment[segment_index[n-1]+1:,:])
			else: segment_list.append(segment)
		else: segment_list.append(segment)
	return segment_list

def find_total_distance(segment):
	total_distance = 0
	for i in range(len(segment)-1):
		total_distance += distance(segment[i],segment[i+1])
	return total_distance

def find_2d_v_t(segment,vmax,amax,t0):
	v = []
	t = []
	total_s = find_total_distance(segment)
	if total_s > vmax**2/amax:
		dist = 0
		s1,s2,t1,t2 = find_key_distance_long(total_s,vmax,amax)
		for i in range(len(segment)-1):
			dist += distance(segment[i],segment[i+1])
			if dist <= s1:
				tmpt_t = np.sqrt(2*dist/amax)
				t.append(t0+tmpt_t)
				v.append(amax*tmpt_t)
			elif dist > s1 and dist <= s2:
				t.append(t0+t1+(dist-s1)/vmax)
				v.append(vmax)
			elif dist > s2:
				tmpt_t = t1+t2-np.sqrt(2*(total_s-dist)/amax)
				t.append(t0+tmpt_t)
				v.append(vmax-amax*(tmpt_t-t2))
	else:
		dist = 0
		s1,t1 = find_key_distance_short(total_s,amax)
		for i in range(len(segment)-1):
			dist += distance(segment[i],segment[i+1])
			if dist <= s1:
				tmpt_t = np.sqrt(2*dist/amax)
				t.append(t0+tmpt_t)
				v.append(amax*tmpt_t)
			else:
				tmpt_t = 2*t1-np.sqrt(2*(total_s-dist)/amax)
				t.append(t0+tmpt_t)
				v.append(amax*(2*t1-tmpt_t))
	return v,t

def decompose_to_xy(segment,v,t):
	vel2d = []
	acc2d = []
	for i in range(len(segment)-1):
		x = segment[i+1][0] - segment[i][0]
		y = segment[i+1][1] - segment[i][1]
		vx = x/np.sqrt(x**2+y**2)*v[i]
		vy = y/np.sqrt(x**2+y**2)*v[i]
		vel2d.append([vx,vy])
		acc2d.append([vx/t[i],vy/t[i]])
	return vel2d,acc2d

def find_3d_v_and_a(segment,v,t):
	vel2d,acc2d = decompose_to_xy(segment,v,t)
	vel3d = [[subvel1,subvel2,0.] for subvel1,subvel2 in vel2d]
	acc3d = [[subacc1,subacc2,0.] for subacc1,subacc2 in acc2d]
	return vel3d,acc3d

def find_v_and_t(segment_list,vmax,amax):
	timestamp = [0]
	vel = [[0,0,0]]
	accel = []
	for segment in segment_list:
		v,t = find_2d_v_t(segment,vmax,amax,timestamp[-1])
		timestamp.extend(t)
		vel3d,acc3d = find_3d_v_and_a(segment,v,t)
		# print(vel3d)
		vel.extend(vel3d)
		accel.extend(acc3d)
	accel.append([0,0,0])
	return timestamp,vel,accel


def add_points(x,y,contact):
	x = np.asarray(x)
	y = np.asarray(y)
	contact = flatten_list(contact)
	coordinates_pair_list = np.array([x,y]).T
	coordinates = []
	# flatten the coordinate list
	for pair in coordinates_pair_list:
		for j in range(len(pair[0])):
			coordinates.append(np.array([pair[0][j],pair[1][j]]))
	path = np.array([coordinates[0]])
	contact_new = [1]
	print("Adding neccessary waypoints...")
	for i in range(len(coordinates)-1):
		dis = distance(coordinates[i],coordinates[i+1])
		line_vec = (coordinates[i+1] - coordinates[i])/dis
		if contact[i] == 1:
			while dis > point_spacing:
				new_point = coordinates[i] + point_spacing*line_vec
				path = np.vstack((path,new_point))
				contact_new.append(1)
				dis = distance(new_point,coordinates[i+1])
				coordinates[i] = new_point
			contact_new.append(1)
		else:
			while dis > point_spacing:
				new_point = coordinates[i] + point_spacing*line_vec
				path = np.vstack((path,new_point))
				contact_new.append(0)
				dis = distance(new_point,coordinates[i+1])
				coordinates[i] = new_point
			contact_new.append(0)
		path = np.vstack((path,coordinates[i+1]))
	return path,contact_new


def append_point(path_3d,path,distance,contactbool,i,contact_new):
	point_temp = np.append(path[i],distance)
	contact_new.append(contactbool)
	path_3d = np.vstack((path_3d,point_temp))
	return path_3d,contact_new


def partial_addition(lst,index,val):
	index += 1
	new_list = lst[:index]
	for item in lst[index:]: new_list.append(item+val)
	return new_list


def generate_3d_trajectory(path,contact,vel,timestamp,acc,t_mid,z_distance,max_vel_z):
	path_3d = np.array([path[0][0],path[0][1],0.0])
	vel3d = [vel[0]]
	timestamps = [0]
	acceleration = [acc[0]]
	time_difference = 2*t_mid
	contact = contact[1:]
	contact.append(1)
	for i in range(1,len(contact)):
		if contact[i-1] == 1 and contact[i] == 1:
			path_3d, contact_new = append_point(path_3d,path,0.0,1,i,contact)
			vel3d.append(vel[i])
			timestamps.append(timestamp[i])
			acceleration.append(acc[i])
		elif contact[i-1] == 1 and contact[i] == 0:
			path_3d, contact_new = append_point(path_3d,path,0.0,1,i,contact)
			vel3d.append(vel[i])
			timestamps.append(timestamp[i])
			acceleration.append([0,0,amax_z])
			path_3d, contact_new = append_point(path_3d,path,z_distance/2,0,i,contact)
			vel3d.append([0,0,max_vel_z])
			timestamps.append(timestamp[i]+t_mid)
			acceleration.append([0,0,-amax_z])
			path_3d, contact_new = append_point(path_3d,path,z_distance,0,i,contact)
			vel3d.append([0,0,0])
			timestamps.append(timestamp[i]+2*t_mid)
			acceleration.append(acc[i])
			timestamp = partial_addition(timestamp,i,time_difference)
			# print(timestamp)
		elif contact[i-1] == 0 and contact[i] == 0:
			path_3d, contact_new = append_point(path_3d,path,z_distance,0,i,contact)
			vel3d.append(vel[i])
			timestamps.append(timestamp[i])
			acceleration.append(acc[i])
		elif contact[i-1] == 0 and contact[i] == 1:
			path_3d, contact_new = append_point(path_3d,path,z_distance,0,i,contact)
			vel3d.append(vel[i])
			timestamps.append(timestamp[i])
			acceleration.append([0,0,-amax_z])
			path_3d, contact_new = append_point(path_3d,path,z_distance/2,0,i,contact)
			vel3d.append([0,0,max_vel_z])
			timestamps.append(timestamp[i]+t_mid)
			acceleration.append([0,0,amax_z])
			path_3d, contact_new = append_point(path_3d,path,0.0,1,i,contact)
			vel3d.append([0,0,0])
			timestamps.append(timestamp[i]+2*t_mid)
			acceleration.append(acc[i])
			timestamp = partial_addition(timestamp,i,time_difference)
			# print(timestamp)
		else:
			print("error occured in contact list")
	return path_3d,vel3d,acceleration,timestamps


def abs_v(v):
	v_val = []
	for vel in v: v_val.append(np.linalg.norm(vel))
	return v_val

def set_axes_equal(ax,path_3d):
    x_limits = [np.amax(path_3d[:,0]),np.amin(path_3d[:,0])]
    y_limits = [np.amax(path_3d[:,1]),np.amin(path_3d[:,1])]
    z_limits = [np.amax(path_3d[:,2]),np.amin(path_3d[:,2])]

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([0, z_middle + plot_radius])

def plot_colourline_3d(x,y,z,c):
    c = cm.plasma(c/np.max(c))
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], [z[i],z[i+1]],c=c[i])
    return


x,y,contact = find_whole_trajectory(list_char)
print("Trajectory calculated.")
path,contact_new = add_points(x,y,contact)
path = path*text_size/880
# plt.scatter(path[:,0],path[:,1])
print("Points added.")
print("Calculating velocity...")
segment_list = find_segment(path,angle_threshold,contact_new)


# find the time taken for drone to rise half way
if z_distance > vmax_z**2/amax:
	t_mid = vmax_z/amax_z + z_distance/(2*vmax_z) - vmax_z/(2*amax_z)
	max_vel_z = vmax_z
else:
	t_mid = np.sqrt(z_distance/amax_z)
	max_vel_z = amax_z*t_mid


timestamp,vel,acc = find_v_and_t(segment_list,vmax,amax)
path_3d,vel3d,acc3d,timestamps = generate_3d_trajectory(path,contact_new,vel,timestamp,acc,t_mid,z_distance,max_vel_z)


while True:
	matplotlib.use('TkAgg')
	animated = input("Enter 1 if you want to see the animation, 0 if you want to see the plot:\n")
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect('equal')
	set_axes_equal(ax,path_3d)
	# animated = "1"
	if animated == "1":
		for i in range(1,len(path_3d)-1):
			start_time = time.time()
			ax.plot(path_3d[i-1:i+1,0],path_3d[i-1:i+1,1],path_3d[i-1:i+1,2],c='m')
			plt.draw()
			sleeper = timestamps[i]-timestamps[i-1]
			time_elapsed = time.time() - start_time
			if sleeper>time_elapsed:
				# plt.pause(sleeper-time_elapsed)
				plt.pause(0.01)
		print("Done!")
		break
	elif animated == "0":
		plot_colourline_3d(path_3d[:,0],path_3d[:,1],path_3d[:,2],abs_v(vel3d[1:]))
		print("Done!")
		break
	else:
		print("Please enter a valid response")
		continue

vel3d = np.asarray(vel3d)
acc3d = np.asarray(acc3d)
timestamps = np.asarray(timestamps).reshape(len(timestamps),1)
timestamps = timestamps*10**9
print(vel3d.shape,acc3d.shape,timestamps.shape,path_3d.shape)
trajectory = np.concatenate((np.concatenate((np.concatenate((path_3d,vel3d),axis=1),acc3d),axis=1),timestamps),axis=1)
np.savetxt("trajectory.txt",trajectory,fmt='%f,%f,%f,%f,%f,%f,%f,%f,%f,%f')

# plot_colourline_3d(path_3d[:,0],path_3d[:,1],path_3d[:,2],abs_v(vel3d[1:]))
plt.show()

