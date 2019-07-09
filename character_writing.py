# get glyph from font

from ttfquery import describe
from ttfquery import glyphquery
import ttfquery.glyph as glyph
import ttfquery
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

animate = False


char = "A"
font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
g = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
contours = g.calculateContours(font)
x = np.array([])
y = np.array([])
outline_x = np.array([])
outline_y = np.array([])


def change_order_append_to_list(outline,points,index,max_index):
	if index == max_index:
		pass
	else:
		n = index +1
		while n < max_index:
			outline.append(points[n])
			n+=1
	for i in range(index):
		outline.append(points[i])
	return outline

a = [2,1,3,4]
b = [5,1,4,2,3]
path = change_order_append_to_list(a,b,b.index(a[-1]),len(b))
print(path)



i = len(contours)
print(i)
# outline=ttfquery.glyph.decomposeOutline(contours[1], steps=5)
# for point in outline:
# 	outline_x = np.append(outline_x, point[0])
# 	outline_y = np.append(outline_y, point[1])
print(contours)
for contour in contours:
	outline = ttfquery.glyph.decomposeOutline(contour, steps=3)
	# outline = sorted(set(outline),key=outline.index)
	print(outline)
	for point in outline:
		outline_x = np.append(outline_x, point[0])
		outline_y = np.append(outline_y, point[1])
	# print(outline)
	for point, flag in contour:
    	# if flag == 1:
		x = np.append(x,point[0])
		y = np.append(y,point[1])

# print(x)
# print(y)
# print(i)

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
	plt.subplot(1,2,1)
	plt.plot(x,y,marker='x')
	plt.subplot(1,2,2)
	plt.plot(outline_x,outline_y,marker='x')
	plt.gca().set_aspect('equal', adjustable='box')

plt.show()