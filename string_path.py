import numpy as np

char = "V"
if char.isupper():
	filename = "letters/char_cap_{}.txt".format(char)
else:
	filename = "letters/char_lower_{}.txt".format(char)

# outline_x,outline_y = np.loadtxt(filename,delimiter="\n")
# print(outline_x)
# print(outline_y)

newf=""
with open(filename,'r') as f:
    for line in f:
        newf+=line.strip()+" 1\n"
    f.close()
with open(filename,'w') as f:
    f.write(newf)
    f.close()


# with open(filename, 'r') as f: 
# 	lines = f.readlines()
# 	x = []
# 	y = []
# 	# contact = []
# 	for line in lines:
# 		x.append(int(line.split(' ')[0]))
# 		y.append(int(line.split(' ')[1]))
# 		# contact.append(line.split(' ')[2])

# print(x)
# print(y)

	
# input_string = input("Enter something for the drone to write: (no punctuation atm) \n")

