import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# to_write = input("Enter something: \n")
to_write = "Hello World"
list_char = list(to_write)
width = 0

font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
two_stroke_char = ["i","j","w","Q","W"]
v = 500
timestp = 0.0


def distance(a,b):
	return np.linalg.norm(np.asarray(a)-np.asarray(b))

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
		plt.plot(x_all[i],y_all[i])

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


x_all,y_all,contact_all =find_whole_trajectory(list_char)
# print(x_all,'\n',y_all)
timestamp = find_timestamp(x_all,y_all)
print(timestamp)
# plot_trajectory(x_all,y_all)
plot_trajectory_with_t(x_all,y_all,timestamp)

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# print(x)
# print(y)

	
# input_string = input("Enter something for the drone to write: (no punctuation atm) \n")

# newf=""
# with open(filename,'r') as f:
#     for line in f:
#         newf+=line.strip()+" 1\n"
#     f.close()
# with open(filename,'w') as f:
#     f.write(newf)
#     f.close()