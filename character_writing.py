# get glyph from font

from ttfquery import describe
from ttfquery import glyphquery
import ttfquery.glyph as glyph
import ttfquery
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx

animate = True
graph = False


char = "R"
font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
glyph = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
contours = glyph.calculateContours(font)
G = nx.Graph()
outline_x = np.array([])
outline_y = np.array([])


def round_all(stuff):
    if isinstance(stuff, list):
        return [round_all(x) for x in stuff]
    if isinstance(stuff, tuple):
        return tuple(round_all(x) for x in stuff)
    return round(float(stuff))

def find_next_path(init_point,path):
	print(init_point,list(G.adj[init_point]))
	next_point = list(G.adj[init_point])[0]
	path.append(eval(next_point))
	G.remove_edge(init_point,next_point)
	return path, next_point

n_contours = len(contours)
print(n_contours)
path = []
if n_contours == 1:
	outline = ttfquery.glyph.decomposeOutline(contours[0], steps=3)
	for points in outline:
		outline_x = np.append(outline_x,points[0])
		outline_y = np.append(outline_y,points[1])
elif n_contours > 1:
	for contour in contours:
		outline = ttfquery.glyph.decomposeOutline(contour, steps=3)
		outline = round_all(outline)
		# outline = sorted(set(outline),key=outline.index)
		outline_str = [str(x) for x in outline]
		for i in range(len(outline)-1):
			G.add_node(outline_str[i])
			G.add_edge(outline_str[i],outline_str[i+1])
	G.remove_edges_from(G.selfloop_edges())
	if graph:
		nx.draw(G,with_labels=True)
		plt.show()
	for p,d in G.degree:
		if d == 1:
			start_point = p
			path.append(eval(p))
			# print(list(G.adj[p]))
			for i in range(G.number_of_edges()):
				if char == "R" and p == str((163,352)):
					G.add_edge(str((163,352)),str((0,352)))
					print("added")
				path,next_point = find_next_path(p,path)
				p = next_point
			break

for i in range(len(path)):
	outline_x = np.append(outline_x,path[i][0])
	outline_y = np.append(outline_y,path[i][1])



if not graph:
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