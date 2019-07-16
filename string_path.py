import numpy as np
from ttfquery import describe
from ttfquery import glyphquery
import matplotlib.pyplot as plt

# to_write = input("Enter something: \n")
to_write = "The quick brown fox jumps over the lazy dog"
list_char = list(to_write)
width = 0

font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
two_stroke_char = ["i","j","w","Q","W"]

def find_trajectory(filename,width):
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
	return x,y

def plot_trajectory(x,y):
	if any(isinstance(ls,list) for ls in x):
		plt.plot(x[0],y[0])
		plt.plot(x[1],y[1])
	else:
		plt.plot(x,y)

for char in list_char:
	# print(char)
	if char.isupper():
		filename = "letters/char_cap_{}.txt".format(char)
		x,y = find_trajectory(filename,width)
		width += glyphquery.width(font,char)
	elif char.islower():
		filename = "letters/char_lower_{}.txt".format(char)
		x,y = find_trajectory(filename,width)
		width += glyphquery.width(font,char)
	else:
		width += 150
	if x is None:
		pass
	else:
		plot_trajectory(x,y)

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