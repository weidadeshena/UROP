import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib.pyplot as plt
import matplotlib.cm as cm


to_write = input("Enter a word: \n")
# to_write = "NaTuRaL wRiTiNg"
list_char = list(to_write)
width = 0
angle_threshold = 130/180*np.pi
point_spacing = 100

font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
# v = 500
timestp = 0.0
space_width = 400
text_size = 1
z_distance = 0.2


amax = 50
vmax = 10


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

def plot_trajectory(x_all,y_all):
	for i in range(len(x_all)):
		plt.plot(x_all[i],y_all[i],marker="o")

def plot_colourline(x,y,c):
    c = cm.plasma([v/vmax for v in c])
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    return

def plot_trajectory_with_t(x_all,y_all,timestamp):
	x_flat = [point for sublist in x_all for point in sublist]
	y_flat = [point for sublist in y_all for point in sublist]
	plot_colourline(x_flat,y_flat,timestamp)

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

def find_v_and_t(segment,vmax,amx,t0):
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

def generate_3d_trajectory(path,contact,timestamp):
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
	trajectory = np.concatenate((np.concatenate((path_3d,contact),axis=1),timestamp),axis=1)
	return trajectory

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


x,y,contact = find_whole_trajectory(list_char)
print("Trajectory calculated.")
path,contact_new = add_points(x,y,contact)
path = path*text_size/880
print("Points added.")
print("Calculating velocity...")
segment_list = find_segment(path,angle_threshold,contact_new)
print(segment_list)


t0=0
v_list = []
t_list = []
for segment in segment_list:
	v,t = find_v_and_t(segment,vmax,amax,t0)
	print(v)
	t0 = t[-1]
	v_list.append(v)
	t_list.append(t)
print("timestamp generated")


t_list_flat = flatten_list(t_list)
t_list_flat.insert(0,0)
v_comparison_list = []
a_comparison_list = []
for i in range(1,len(path)-1):
	velocity = distance(path[i-1],path[i+1])/2*(t_list_flat[i+1]-t_list_flat[i-1])
	acceleration = velocity/2*(t_list_flat[i+1]-t_list_flat[i-1])
	v_bool = abs(velocity)<vmax
	a_bool = abs(acceleration)<amax
	v_comparison_list.append(v_bool)
	a_comparison_list.append(a_bool)




if all(v_comparison_list) and all(a_comparison_list):
	print("sanity check: \nvelocity and acceleration is below the max value \nproceeding...")
	plt.gca().set_aspect('equal', adjustable='box')
	plt.xlim(np.amin(path[:,0])-1,np.amax(path[:,0])+1)
	plt.ylim(np.amin(path[:,1])-1,np.amax(path[:,1])+1)
	k=0
	while True:
		animated = input("Enter 1 if you want to see the animation, 0 if you want to see the plot:\n")
		# animated = "1"
		if animated == "1":
			for i in range(len(segment_list)):
				k+=1
				for j in range(1,len(segment_list[i])-1):
					if contact_new[k] == 1:
						x_coord = segment_list[i][j-1:j+1,0].T
						y_coord = segment_list[i][j-1:j+1,1].T
						plt.plot(x_coord,y_coord,c='r')
						plt.draw()
						if (t_list[i][j]-t_list[i][j-1])> 0.001:
							plt.pause(t_list[i][j]-t_list[i][j-1])
					else:
						x_coord = segment_list[i][j-1:j+1,0].T
						y_coord = segment_list[i][j-1:j+1,1].T
						plt.plot(x_coord,y_coord,c='b')
						plt.draw()
						if (t_list[i][j]-t_list[i][j-1])> 0.001:
							plt.pause(t_list[i][j]-t_list[i][j-1])
					k+=1
				if contact_new[k] == 1:
					plt.plot(segment_list[i][-2:,0],segment_list[i][-2:,1],c='r')
				else:
					plt.plot(segment_list[i][-2:,0],segment_list[i][-2:,1],c='b')
			print("Done!")
			break
		elif animated == "0":
			for i in range(len(segment_list)):
				plot_colourline(segment_list[i][:,0].T,segment_list[i][:,1].T,v_list[i])
			print("Done!")
			break
		else:
			print("Please enter a valid response")
			continue
	contact = np.array([contact_new]).T
	t_list = np.insert(np.array(sorted(flatten_list(t_list))),0,0).T.reshape(len(contact),1)
	# trajectory = np.concatenate((np.concatenate((path,contact),axis=1),t_list),axis=1)
	trajectory = generate_3d_trajectory(path,contact,t_list)
	# print(trajectory)
	# np.savetxt("trajectory.txt",trajectory,fmt='%d %d %d %f')
	plt.show()
else:
	print("this is not safe your math is wrong you suck")










# newf=""
# with open(filename,'r') as f:
#     for line in f:
#         newf+=line.strip()+" 1\n"
#     f.close()
# with open(filename,'w') as f:
#     f.write(newf)
#     f.close()


