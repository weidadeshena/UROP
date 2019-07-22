#add_1

char = "~"
filename = "letters/{}.txt".format(ord(char))
newf=""
with open(filename,'r') as f:
    for line in f:
        newf+=line.strip()+" 1\n"
    f.close()
with open(filename,'w') as f:
    f.write(newf)
    f.close()
