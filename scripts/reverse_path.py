# reverse some part of path
import numpy as np

char = "R"

filename = "letters/{}.txt".format(ord(char))


x = []
y = []
contact = []
with open(filename, 'r') as f: 
		lines = f.readlines()
		for line in lines:
			x.append(int(line.split(' ')[0]))
			y.append(int(line.split(' ')[1]))
			contact.append(int(line.split(' ')[2]))

def reverse_second_part(x,index):
	x1 = x[:index]
	x2 = x[index:]
	x2.reverse()
	for point in x2:
		x1.append(point)
	return x1

def reverse_first_part(x,index):
	x1 = x[:index]
	x2 = x[index:]
	x1.reverse()
	for point in x2:
		x1.append(point)
	return x1

reverse_index = len(x)-5

x1 = reverse_first_part(x,reverse_index)
y1 = reverse_first_part(y,reverse_index)

content = np.array([x1,y1,contact]).T
print(content)

np.savetxt("letters/{}.txt".format(ord(char)),content,fmt=['%d','%d','%d'])



