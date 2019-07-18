import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# to_write = input("Enter something: \n")
to_write = "A"
list_char = list(to_write)
width = 0
angle_threshold = 120/180*np.pi

font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
two_stroke_char = ["i","j","w","Q","W"]
v = 500
timestp = 0.0

amax = 50
vmax = 100

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

def find_char_trajectory(char,width):
	x = []
	y = []
	contact = []
	if char.isupper():
		filename = "letters/char_cap_{}.txt".format(char)
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
	elif char.islower():
		filename = "letters/char_lower_{}.txt".format(char)
		with open(filename, 'r') as f: 
			lines = f.readlines()
			x = []
			y = []
			contact = []
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

def find_whole_trajectory(list_char):
	x_all = []
	y_all = []
	contact_all = []
	global width
	for char in list_char:
		if char.isalpha():
			x,y,contact = find_char_trajectory(char,width)
			width += glyphquery.width(font,char)
			contact_all.append(contact)
			if any(isinstance(ls,list) for ls in x):
				x_all.append(x[0])
				x_all.append(x[1])
				y_all.append(y[0])
				y_all.append(y[1])
			else:
				x_all.append(x)
				y_all.append(y)
		else: width += 150
	return x_all,y_all,contact_all

def plot_trajectory(x_all,y_all):
	for i in range(len(x_all)):
		plt.plot(x_all[i],y_all[i],marker="o")

def plot_colourline(x,y,c):
    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
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
	segment_index = np.argwhere(angle_array<angle_threshold)
	segment_index = segment_index.T
	segment_list = []
	segment_list.append(path[0:segment_index[0][0]+2,:])
	segment_list.append(path[segment_index[0][0]+1:segment_index[0][1]+2,:])
	n = segment_index.shape[1]
	for i in range(1,n-1):
		segment_list.append(path[segment_index[0][i]+1:segment_index[0][i+1]+2,:])
	segment_list.append(path[segment_index[0][n-1]+1:,:])
	return segment_list

x,y,contact = find_char_trajectory("B",0)
x = np.asarray(x)
y = np.asarray(y)
coordinates = np.array([x,y]).T
# for i in range(len(x)-2):
# 	# print(coordinates[i],coordinates[i+1])
# 	# print(type(coordinates[i]))
# 	angle = find_angle(coordinates[i],coordinates[i+1],coordinates[i+2])
# 	print(coordinates[i])
# 	print(angle)

path = np.array([coordinates[0]])
for i in range(len(x)-1):
	# print(i)
	dis = distance(coordinates[i],coordinates[i+1])
	line_vec = (coordinates[i+1] - coordinates[i])/dis
	while dis > 85:
		# print(dis)
		new_point = coordinates[i] + 85*line_vec
		path = np.vstack((path,new_point))
		dis = distance(new_point,coordinates[i+1])
		coordinates[i] = new_point
	path = np.vstack((path,coordinates[i+1]))
# print(path)
# plot_trajectory(path.T[0],path.T[1])
segment_list = find_segment(path,angle_threshold)
for segment in segment_list



# np.savetxt("letters/cap_B.txt",coordinates,fmt='%d %d')

# x_all,y_all,contact_all =find_whole_trajectory(list_char)
# # print(x_all,'\n',y_all)
# timestamp = find_timestamp(x_all,y_all)
# print(timestamp)
# # plot_trajectory(x_all,y_all)
# plot_trajectory_with_t(x_all,y_all,timestamp)

# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()




newf=""
# with open(filename,'r') as f:
#     for line in f:
#         newf+=line.strip()+" 1\n"
#     f.close()
# with open(filename,'w') as f:
#     f.write(newf)
#     f.close


# newf=""
# with open(filename,'r') as f:
#     for line in f:
#         newf+=line.strip()+" 1\n"
#     f.close()
# with open(filename,'w') as f:
#     f.write(newf)
#     f.close()


