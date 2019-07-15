import numpy as np

char = "a"
if char.isupper():
	filename = "letters/char_cap_{}.txt".format(char)
else:
	filename = "letters/char_lower_{}.txt".format(char)

outline_x,outline_y = np.loadtxt(filename,delimiter="\n")
print(outline_x)
print(outline_y)


# input_string = input("Enter something for the drone to write: (no punctuation atm) \n")

