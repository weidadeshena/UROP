import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


to_write = input("Enter a word: \n")
# to_write = "NaTuRaL wRiTiNg"
list_char = list(to_write)
width = 0
angle_threshold = 130/180*np.pi
point_spacing = 50
font_url = "fonts/cnc_v.ttf"
font = describe.openFont(font_url)
# v = 500
timestp = 0.0
space_width = 400
text_size = 880
z_distance = 0.2

amax_x = 100
amax_y = 500
amax_z = 100

vmax_x = 10
vmax_y = 50
vmax_z = 10
vmax = np.sqrt(vmax_x**2+vmax_y**2+vmax_z**2)

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
def find_whole_trajectory_2d(list_char):
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
	x_all = flatten_list(x_all)
	y_all = flatten_list(y_all)
	contact_all = flatten_list(contact_all)
	return x_all,y_all,contact_all


def plot_colourline(x,y,z,c):
    c = cm.plasma(c/np.max(c))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], [z[i],z[i+1]],c=c[i])
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

def find_total_distance(segment):
	total_distance = 0
	for i in range(len(segment)-1):
		total_distance += abs(segment[i+1]-segment[i])
	return total_distance

# add a point every point_spacing unit
def add_points(x,y,contact):
	x = np.asarray(x)/880*text_size
	y = np.asarray(y)/880*text_size
	coordinates = np.array([x,y]).T
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


def find_3d_path(path,contact):
	path_3d = np.array([path[0][0],path[0][1],0.0])
	for i in range(1,len(path)):
		if contact[i-1] == 1:
			point_temp = np.append(path[i],0.0)
			path_3d = np.vstack((path_3d,point_temp))
		elif contact[i-1] == 0 and contact[i] == 0:
			point_temp = np.append(path[i],z_distance)
			path_3d = np.vstack((path_3d,point_temp))
		elif contact[i-1] == 0 and contact[i] == 1:
			point_temp = np.append(path[i],0.0)
			path_3d = np.vstack((path_3d,point_temp))
		else:
			print("error occured in contact list")
	return path_3d


def find_segment(path_1d): # for x and y
	dot_product_array = np.array([])
	for i in range(len(path_1d)-2):
		dot_product = (path_1d[i+2]-path_1d[i+1])*(path_1d[i+1]-path_1d[i])
		dot_product_array = np.append(dot_product_array,dot_product)
	segment_index = np.argwhere(dot_product_array<0)
	# print(segment_index)
	segment_list = []
	segment_list.append(path_1d[:segment_index[0][0]+2])
	# segment_list.append(path[segment_index[0][0]+1:segment_index[0][1]+2,:])
	n = len(segment_index)
	for i in range(n-1):
		segment_list.append(path_1d[segment_index[i][0]+1:segment_index[i+1][0]+2])
	segment_list.append(path_1d[segment_index[n-1][0]+1:])
	# segment_list = indexsplit(1dpath,segment_index)
	return segment_list

def find_segment_z(path_z): # for z direction only:
	difference_array = np.array([])
	for i in range(len(path_z)-1):
		difference = path_z[i+1] - path_z[i]
		difference_array = np.append(difference_array,difference)
	segment_index = np.argwhere(abs(difference_array)>0.1).T[0]
	segment_list = []
	segment_list.append(path_z[:segment_index[0]+1])
	segment_list.append(path_z[segment_index[0]:segment_index[0]+2])
	# segment_list.append(path[segment_index[0][0]+1:segment_index[0][1]+2,:])
	n = len(segment_index)
	for i in range(n-1):
		segment_list.append(path_z[segment_index[i]+1:segment_index[i+1]+1])
		segment_list.append(path_z[segment_index[i+1]:segment_index[i+1]+2])
	segment_list.append(path_z[segment_index[n-1]+1:])
	# segment_list = indexsplit(1dpath,segment_index)
	return segment_list


def find_t(segment_list,vmax,amax):
	t = [0]
	v = []
	for segment in segment_list:
		t0 = t[-1]
		total_s = abs(find_total_distance(segment))
		# if total_s > vmax**2/amax:
		# 	dist = 0
		# 	s1,s2,t1,t2 = find_key_distance_long(total_s,vmax,amax)
		# 	for i in range(len(segment)-1):
		# 		dist += distance(segment[i],segment[i+1])
		# 		if dist <= s1:
		# 			tmpt_t = np.sqrt(2*dist/amax)
		# 			t.append(t0+tmpt_t)
		# 		elif dist > s1 and dist <= s2:
		# 			t.append(t0+t1+(dist-s1)/vmax)
		# 		elif dist > s2:
		# 			tmpt_t = t1+t2-np.sqrt(2*(total_s-dist)/amax)
		# 			t.append(t0+tmpt_t)

		# else:
		# 	dist = 0
		# 	s1,t1 = find_key_distance_short(total_s,amax)
		# 	for i in range(len(segment)-1):
		# 		dist += distance(segment[i],segment[i+1])
		# 		if dist <= s1:
		# 			tmpt_t = np.sqrt(2*dist/amax)
		# 			t.append(t0+tmpt_t)
		# 		else:
		# 			tmpt_t = 2*t1-np.sqrt(2*(total_s-dist)/amax)
		# 			t.append(t0+tmpt_t)
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
	t = np.asarray(t)
	print(t,'\n',v)
	return t

# take the biggest time stamp and calculate v on the other two direction based on that
def compensating_for_something(path_3d):
	global vmax_y,vmax_x,vmax_z,amax_z,amax_x,amax_y
	v_and_a = [[vmax_x,amax_x],[vmax_y,amax_y],[vmax_z,amax_z]] 
	# lol the same name as the museum... I'll see myself out
	timestamps=[0]

	# for i in range(3):
	# 	if i != 2:
	# 		segment_list = find_segment(path_3d[:,i].T)
	# 	else:
	# 		segment_list = find_segment_z(path_3d[:,i].T)
	# 	t = find_t(segment_list,v_and_a[i][0],v_and_a[i][1])
	# 	t_3d.append(t)
	# t_3d = np.asarray(t_3d).T
	# print(t_3d)
	# v_3d = []
	# a_3d = []
	# timestamps = [0]
	# n = len(t_3d)
	# for i in range(1,n):
	# 	t = np.amax(t_3d[i,:])
	# 	del_t = t-timestamps[-1]
	# 	timestamps.append(t)
	# 	v_temp = []
	# 	a_temp = []
	# 	if (n-i) < 0.1:
	# 		for j in range(3):
	# 			v = (path_3d[i][j]-path_3d[i-1][j])/(del_t)
	# 			v_temp.append(v)
	# 			a_temp.append(v/del_t)
	# 	else:
	# 		print(i)
	# 		for j in range(3):
	# 			v = (path_3d[i+1][j]-path_3d[i-1][j])/(2*del_t)
	# 			v_temp.append(v)
	# 			a_temp.append(v/del_t)
	# 	v_temp = np.asarray(v_temp)
	# 	a_temp = np.asarray(a_temp)
	# 	v_3d.append(v_temp)
	# 	a_3d.append(a_temp)
	v_3d = []
	a_3d = []
	v = [0,0,0]
	for j in range(len(path_3d)-1):
		t = []
		for i in range(3):
			# print(abs(v[i]))
			# print(v_and_a[i][0])
			print(i)
			if path_3d[j+1,i]-path_3d[j,i] > 0:
				if abs(v[i]) < v_and_a[i][0]:
					delt = (-v[i]+np.sqrt(v[i]**2+2*v_and_a[i][1]*(path_3d[j+1,i]-path_3d[j,i])))/v_and_a[i][1]
					t.append(delt)
					# print(t)
				else:
					delt = (path_3d[j+1,i]-path_3d[j,i])/v_and_a[i][0]
					t.append(delt)
					# print(t)
			else:
				if abs(v[i]) < v_and_a[i][0]:
					delt = (-v[i]+np.sqrt(v[i]**2+2*-v_and_a[i][1]*(path_3d[j+1,i]-path_3d[j,i])))/v_and_a[i][1]
					t.append(delt)
					# print(t)
				else:
					delt = (path_3d[j+1,i]-path_3d[j,i])/(-v_and_a[i][0])
					t.append(delt)
					# print(t)
		# print(t)
		del_t = max(t)

		timestamps.append(del_t+timestamps[-1])
		v_temp = []
		a_temp = []
		for i in range(3):
			vel = (path_3d[j+1,i]-path_3d[j,i])/del_t
			v_temp.append(vel)
			a_temp.append(vel/del_t)
		v = v_temp
		v_3d.append(v_temp)
		a_3d.append(a_temp)
	v_3d.append([0.,0.,0.])
	a_3d.append([0.,0.,0.])
	return v_3d,a_3d,timestamps




		



	v_3d.append(np.array([0.0,0.0,0.0]))
	a_3d.append(np.array([0.0,0.0,0.0]))
	return v_3d,a_3d,timestamps


def modulus_of_each_subarray(arry):
	modulus = np.array([])
	for i in range(len(arry)):
		modulus = np.append(modulus,np.linalg.norm(arry[i]))
	return modulus




x,y,contact = find_whole_trajectory_2d(list_char)
print("2D trajectory calculated.")
# print(x,y)


path,contact_new = add_points(x,y,contact)
print("Points added.")
path_3d = find_3d_path(path,contact_new)
print("Trajectory is converted to 3D")
v_3d,a_3d,t_3d = compensating_for_something(path_3d)
total_velocity = modulus_of_each_subarray(v_3d)


print(v_3d)
print(t_3d)

plot_colourline(path_3d[:,0].T,path_3d[:,1].T,path_3d[:,2].T,total_velocity)
plt.show()


# # add points in gap
# path,contact_new = add_points(x,y,contact)
# path = path*text_size/880
# path_3d = find_3d_path(path,contact_new)
# print("Points added.")
# print("generating timestamps...")
# segments = find_segment(path,angle_threshold)
# segment_list = list()
# for segment in segment_list:
# 	segment_list.append(segment)

# t0=0
# v_list = []
# t_list = []
# for segment in segment_list:
# 	v,t = find_v_and_t(segment,vmax,amax,t0)
# 	t0 = t[-1]
# 	v_list.append(v)
# 	t_list.append(t)
# print("timestamp generated")


# # test if it exceed the limit vmax and amax
# t_list_flat = flatten_list(t_list)
# t_list_flat.insert(0,0)
# v_comparison_list = []
# a_comparison_list = []
# for i in range(1,len(path)-1):
# 	velocity = distance(path[i-1],path[i+1])/2*(t_list_flat[i+1]-t_list_flat[i-1])
# 	acceleration = velocity/2*(t_list_flat[i+1]-t_list_flat[i-1])
# 	v_bool = abs(velocity)<vmax
# 	a_bool = abs(acceleration)<amax
# 	v_comparison_list.append(v_bool)
# 	a_comparison_list.append(a_bool)




# if all(v_comparison_list) and all(a_comparison_list):
# 	print("sanity check: \nvelocity and acceleration is below the max value \nproceeding...")
# 	plt.gca().set_aspect('equal', adjustable='box')
# 	plt.xlim(np.amin(path[:,0])-1,np.amax(path[:,0])+1)
# 	plt.ylim(np.amin(path[:,1])-1,np.amax(path[:,1])+1)
# 	k=0
# 	while True:
# 		animated = input("Enter 1 if you want to see the animation, 0 if you want to see the plot:\n")
# 		# animated = "1"
# 		if animated == "1":
# 			for i in range(len(segment_list)):
# 				k+=1
# 				for j in range(1,len(segment_list[i])-1):
# 					if contact_new[k] == 1:
# 						x_coord = segment_list[i][j-1:j+1,0].T
# 						y_coord = segment_list[i][j-1:j+1,1].T
# 						plt.plot(x_coord,y_coord,c='r')
# 						plt.draw()
# 						if (t_list[i][j]-t_list[i][j-1])> 0.001:
# 							plt.pause(t_list[i][j]-t_list[i][j-1])
# 					else:
# 						x_coord = segment_list[i][j-1:j+1,0].T
# 						y_coord = segment_list[i][j-1:j+1,1].T
# 						plt.plot(x_coord,y_coord,c='b')
# 						plt.draw()
# 						if (t_list[i][j]-t_list[i][j-1])> 0.001:
# 							plt.pause(t_list[i][j]-t_list[i][j-1])
# 					k+=1
# 				if contact_new[k] == 1:
# 					plt.plot(segment_list[i][-2:,0],segment_list[i][-2:,1],c='r')
# 				else:
# 					plt.plot(segment_list[i][-2:,0],segment_list[i][-2:,1],c='b')
# 			print("Done!")
# 			break
# 		elif animated == "0":
# 			for i in range(len(segment_list)):
# 				plot_colourline(segment_list[i][:,0].T,segment_list[i][:,1].T,v_list[i])
# 			print("Done!")
# 			break
# 		else:
# 			print("Please enter a valid response")
# 			continue
# 	contact = np.array([contact_new]).T
# 	t_list = np.insert(np.array(sorted(flatten_list(t_list))),0,0).T.reshape(len(contact),1)
# 	# trajectory = np.concatenate((np.concatenate((path,contact),axis=1),t_list),axis=1)
# 	trajectory = generate_3d_trajectory(path,contact,t_list)
# 	print(trajectory)
# 	# np.savetxt("trajectory.txt",trajectory,fmt='%d %d %d %f')
# 	plt.show()
# else:
# 	print("this is not safe your math is wrong you piece of shit")