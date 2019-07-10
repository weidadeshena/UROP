# get glyph from font

from ttfquery import describe
from ttfquery import glyphquery
import ttfquery.glyph as glyph
import ttfquery
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
# import networkx as nx

animate = True


char = "Q"
font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
g = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
contours = g.calculateContours(font)
x = np.array([])
y = np.array([])
outline_x = np.array([])
outline_y = np.array([])


def change_order_append_to_list(outline,points,index):
	max_index = len(points)
	if index == max_index:
		pass
	else:
		n = index +1
		while n < max_index:
			outline.append(points[n])
			n+=1
	for i in range(index+1):
		outline.append(points[i])
	path = outline
	return path

# a = [(1,2),(2,3),(4,5)]
# b = [(3,5),(4,5),(1,4),(3,6)]
# k = change_order_append_to_list(a,b,b.index(a[-1]),len(b))
# print(k)


n_contours = len(contours)
print(n_contours)
# outline=ttfquery.glyph.decomposeOutline(contours[1], steps=5)
# for point in outline:
# 	outline_x = np.append(outline_x, point[0])
# 	outline_y = np.append(outline_y, point[1])
outline_list = []
for contour in contours:
	outline = ttfquery.glyph.decomposeOutline(contour, steps=3)
	outline = sorted(set(outline),key=outline.index)
	outline_list.append(outline)
	# for point in outline:
	# 	outline_x = np.append(outline_x, point[0])
	# 	outline_y = np.append(outline_y, point[1])

	# for point, flag in contour:
	# 	x = np.append(x,point[0])
	# 	y = np.append(y,point[1])

path = outline_list[0]
for i in range(1,len(outline_list)):
	current_outline = outline_list[i]
	if path[-1] in current_outline:
		path = change_order_append_to_list(path,current_outline,current_outline.index(path[-1]))
	else:
		smallest = np.inf
		for j in range(len(current_outline)):
			difference = np.subtract(path[-1],current_outline[j])
			dist = np.sum(np.absolute(difference))
			if dist < smallest:
				smallest = dist
				min_index = j
		path = change_order_append_to_list(path,current_outline,min_index)


# path = sorted(set(path),key=path.index)
for point in path:
	outline_x = np.append(outline_x, point[0])
	outline_y = np.append(outline_y, point[1])



if animate:
	fig = plt.figure()
	plt.xlim(-200,1000)
	plt.ylim(-200,1000)
	# plt.gca().set_aspect('equal', adjustable='box')
	line, = plt.plot([], [], 'o')
	points = plt.scatter(outline_x[0],outline_y[0],marker='x')

	def animate(i):
		global line,points
		points.remove()
		points = plt.scatter(outline_x[i],outline_y[i],marker='x')
		line = plt.plot(outline_x[:i+1], outline_y[:i+1],'orange')
		# points.remove()
		# return line

	ani = FuncAnimation(fig, animate, frames=100, interval=200)
else:
	plt.plot(outline_x,outline_y,marker='x')
	plt.gca().set_aspect('equal', adjustable='box')

plt.show()