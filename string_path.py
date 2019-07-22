import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# to_write = input("Enter something: \n")
to_write = "#Stefan"
list_char = list(to_write)
width = 0
angle_threshold = 130/180*np.pi
point_spacing = 10

font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
two_stroke_char = ["i","j","w","Q","W"]
v = 500
timestp = 0.0
space_width = 400

amax = 5000
vmax = 2000

# find curvature using Menger curvature and Heron's formula
def find_curvature(point1,point2,point3):
	a = distance(point1,point2)
	b = distance(point2,point3)
	c = distance(point3,point1)
	s = (a+b+c)/2
	curvature = 4*np.sqrt(s*(s-a)*(s-b)*(s-c))/(a*b*c)
	return curvature

def find_angle(point1,point2,point3):
	a = distance(point1,point2)
	b = distance(point2,point3)
	c = distance(point3,point1)
	angle = np.arccos((a**2+b**2-c**2)/(2*a*b))
	return angle

def distance(a,b):
	return np.linalg.norm(a-b)

def find_0(contact):
	return [i for i, x in enumerate(contact) if x == 0]

# constant velocity
def find_timestamp(x,y):
	global timestp
	x = [point for sublist in x for point in sublist]
	y = [point for sublist in y for point in sublist]
	x = np.asarray(x)
	y = np.asarray(y)
	coordinates = np.array([x,y]).T
	timestamp = [timestp]
	for i in range(len(x)-1):
		# print(coordinates[i],coordinates[i+1])
		# print(type(coordinates[i]))
		d = distance(coordinates[i],coordinates[i+1])
		dt = d/v
		timestp += dt
		timestamp.append(timestp)
	return timestamp

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
		# print(contact)
	x = [element+width for element in x]
	if char in two_stroke_char:
		i = contact.index(0)+1
		x1 = x[:i]
		x2 = x[i:]
		y1 = y[:i]
		y2 = y[i:]
		x = [x1,x2]
		y = [y1,y2]
	return x,y,contact

# find the trajectory of the whole word
def find_whole_trajectory(list_char):
	x_all = []
	y_all = []
	contact_all = []
	global width
	for char in list_char:
		if char.isspace():
			width += space_width
		else:
			x,y,contact = find_char_trajectory(char,width)
			glyph_name = glyphquery.glyphName(font,char)
			width += glyphquery.width(font,glyph_name)
			contact_all.append(contact)
			if any(isinstance(ls,list) for ls in x):
				x_all.append(x[0])
				x_all.append(x[1])
				y_all.append(y[0])
				y_all.append(y[1])
			else:
				x_all.append(x)
				y_all.append(y)
	return x_all,y_all,contact_all

def plot_trajectory(x_all,y_all):
	for i in range(len(x_all)):
		plt.plot(x_all[i],y_all[i],marker="o")

def plot_colourline(x,y,c):
    c = cm.plasma([v/2000 for v in c])
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    return

def plot_trajectory_with_t(x_all,y_all,timestamp):
	x_flat = [point for sublist in x_all for point in sublist]
	y_flat = [point for sublist in y_all for point in sublist]
	plot_colourline(x_flat,y_flat,timestamp)
	# print(timestamp)

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

def find_segment(path,angle_threshold):
	angle_array = np.array([])
	for i in range(path.shape[0]-2):
		angle = find_angle(path[i],path[i+1],path[i+2])
		angle_array = np.append(angle_array,angle)
	print(angle_array)
	segment_index = np.argwhere(angle_array<angle_threshold)
	segment_index = segment_index.T
	# print("segment_index: ",segment_index)
	segment_list = []
	segment_list.append(path[0:segment_index[0][0]+2,:])
	segment_list.append(path[segment_index[0][0]+1:segment_index[0][1]+2,:])
	n = segment_index.shape[1]
	for i in range(1,n-1):
		segment_list.append(path[segment_index[0][i]+1:segment_index[0][i+1]+2,:])
	segment_list.append(path[segment_index[0][n-1]+1:,:])
	return segment_list

def find_total_distance(segment):
	total_distance = 0
	for i in range(len(segment)-1):
		total_distance += distance(segment[i],segment[i+1])
	return total_distance

def find_v_and_t(segment,vmax,amx,t0):
	v = [0]
	t = [t0]
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
			elif dist > s2 and dist < total_s:
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




x,y,contact = find_whole_trajectory(list_char)
x = np.asarray(x)
y = np.asarray(y)
print(contact)
coordinates_pair_list = np.array([x,y]).T
coordinates = []
# print(coordinates_pair_list)
for pair in coordinates_pair_list:
	# print(pair)
	for j in range(len(pair[0])):
		coordinates.append(np.array([pair[0][j],pair[1][j]]))

# print(coordinates)


path = np.array([coordinates[0]])
for i in range(len(coordinates)-1):
	# print(i)
	dis = distance(coordinates[i],coordinates[i+1])
	line_vec = (coordinates[i+1] - coordinates[i])/dis
	while dis > point_spacing:
		# print(dis)
		new_point = coordinates[i] + point_spacing*line_vec
		path = np.vstack((path,new_point))
		dis = distance(new_point,coordinates[i+1])
		coordinates[i] = new_point
	path = np.vstack((path,coordinates[i+1]))
# print(path)
# plot_trajectory(path.T[0],path.T[1])
segment_list = find_segment(path,angle_threshold)
t0=0
v_list = []
t_list = []
for segment in segment_list:
	v,t = find_v_and_t(segment,vmax,amax,t0)
	t0 = t[-1]
	v_list.append(v)
	t_list.append(t)

# print(segment_list)

# plt.gca().set_aspect('equal', adjustable='box')

# animation
plt.xlim(-100,np.amax(path)+100)
plt.ylim(-200,800)
for i in range(len(segment_list)):
	try:
		text
	except NameError:
		pass
	else:
		text.remove()
	plot_colourline(segment_list[i][:,0].T,segment_list[i][:,1].T,v_list[i])
	# plt.scatter(segment_list[i][0][0],segment_list[i][0][1])
	# text = plt.text(500,700,"velocity: {}".format(int(v_list[i][0])))
	# plt.draw()
	# for j in range(1,len(t_list[i])):
	# 	text.remove()
	# 	x_coord = segment_list[i][:j,0].T
	# 	y_coord = segment_list[i][:j,1].T
	# 	plt.scatter(x_coord,y_coord)
	# 	text = plt.text(500,700,"velocity: {}".format(int(v_list[i][j])))
	# 	plt.draw()
	# 	plt.pause(t_list[i][j]-t_list[i][j-1])

# plot with colour changes as velocity
# for i in range(len(segment_list)):
# 	plot_colourline(segment_list[i][:,0].T,segment_list[i][:,1].T,v_list[i])



plt.show()

# np.savetxt("letters/cap_B.txt",coordinates,fmt='%d %d')

# x_all,y_all,contact_all =find_whole_trajectory(list_char)
# # print(x_all,'\n',y_all)
# timestamp = find_timestamp(x_all,y_all)
# print(timestamp)
# # plot_trajectory(x_all,y_all)
# plot_trajectory_with_t(x_all,y_all,timestamp)

# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()






# newf=""
# with open(filename,'r') as f:
#     for line in f:
#         newf+=line.strip()+" 1\n"
#     f.close()
# with open(filename,'w') as f:
#     f.write(newf)
#     f.close()


