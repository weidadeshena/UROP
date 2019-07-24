#add_1

# char = "~"
# filename = "letters/{}.txt".format(ord(char))
# newf=""
# with open(filename,'r') as f:
#     for line in f:
#         newf+=line.strip()+" 1\n"
#     f.close()
# with open(filename,'w') as f:
#     f.write(newf)
#     f.close()

def lindexsplit(some_list, indices):
    # Checks to see if any extra arguments were passed. If so,
    # prepend the 0th index and append the final index of the 
    # passed list. This saves from having to check for the beginning
    # and end of args in the for-loop. Also, increment each value in 
    # args to get the desired behavior.
    # if args:
    #     args = (0,) + tuple(data+1 for data in args) + (len(some_list)+1,)
    #     # args.insert(0,0)
    # print(args)
    indices = [i+1 for i in indices]
    indices.insert(0,0)
    indices.append(len(some_list))
    # For a little more brevity, here is the list comprehension of the following
    # statements:
    #    return [some_list[start:end] for start, end in zip(args, args[1:])]
    my_list = []
    print(tuple(zip(indices, indices[1:])))
    for start, end in zip(indices, indices[1:]):
        my_list.append(some_list[start:end])
    return my_list

print(lindexsplit([1,2,3,4,5,6,7,4,3,2,5],[2,4,7]))
# print(tuple(zip([0,3,5,8,12],[3,5,8,12])))