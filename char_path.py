# get glyph from font

from ttfquery import describe
from ttfquery import glyphquery
import ttfquery.glyph as glyph
import ttfquery
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx
from math import floor
animate = True
# draw_graph = True
draw_graph = False

two_strokes = ["Q","W","i","j","w","?"]
only_contour_needed = ["E","F","H","K","T","X","Y","x","f","k","t","x"]


char = "e"
font_url = "cnc_v.ttf"
font = describe.openFont(font_url)
glyph = glyph.Glyph(ttfquery.glyphquery.glyphName(font, char))
contours = glyph.calculateContours(font)
# initialise undirected graph
G = nx.Graph()
# initialise path coordinate array


# round the tuple to a integer so that it will be on the same node
def round_all(stuff):
    if isinstance(stuff, list):
        return [round_all(x) for x in stuff]
    if isinstance(stuff, tuple):
        return tuple(round_all(x) for x in stuff)
    return round(float(stuff))

# find the path with the graph given
def find_next_path(init_point,path):
	print(init_point,sorted(list(G.adj[init_point]),reverse=True))
	if list(G.adj[init_point]):
		next_point = sorted(list(G.adj[init_point]),reverse=True)[0]
		if char == "a" and init_point == str((346,77)):
			next_point = str((344,66))
		if char == "b" and init_point == str((0,700)):
			next_point = str((0,0))
		if char == "h" and init_point == str((0,0)):
			next_point = str((0,690))
		if char == "m" and init_point == str((346,433)):
			if len(list(G.adj[init_point])) == 2:
				next_point = next_point = sorted(list(G.adj[init_point]),reverse=True)[1]
			else:
				pass
		path.append(eval(next_point))
		if G.has_edge(init_point,next_point):
			G.remove_edge(init_point,next_point)
	return path, next_point

# find the top point coordinate in a graph
def find_top():
	nodes = list(G.nodes)
	top_y = 0
	for i in range(len(nodes)):
		if eval(nodes[i])[1] > top_y:
			top_y = eval(nodes[i])[1]
			top_point = nodes[i]
	return top_point # string type

def path_for_easy_char(contour,outline_x,outline_y):
	outline = ttfquery.glyph.decomposeOutline(contour, steps=5)
	outline = round_all(outline)
	# outline.reverse()
	for points in outline:
		outline_x = np.append(outline_x,points[0])
		outline_y = np.append(outline_y,points[1])
	return outline_x,outline_y

def graph_theory_init(contours):
	global G
	for contour in contours:
		outline = ttfquery.glyph.decomposeOutline(contour, steps=5)
		outline = round_all(outline)
		# # char A is a bit special... need to get rid of duplicate points in contours
		if char == "A" or char == "P" or char == "p" or char == "q":
			outline = sorted(set(outline),key=outline.index)
		outline_str = [str(x) for x in outline]
		# set up the undirected graph with coordinates of outline points as nodes
		# if the two points on the contour is connected, an edge is added
		for i in range(len(outline)-1):
			G.add_node(outline_str[i])
			G.add_edge(outline_str[i],outline_str[i+1])
	if char == "d":
		G = nx.contracted_nodes(G,str((38,29)),str((38,28)))
	if char == "p":
		G = nx.contracted_nodes(G,str((308,28)),str((308,29)))
	if char == "h":
		G.add_edge(str((0,0)),str((0,690)))
		G.remove_edge(str((0,0)),str((0,433)))
	if char == "m" or char == "n" or char == "r":
		G.remove_edge(str((0,0)),str((0,433)))
		G.add_edge(str((0,0)),str((0,510)))
	if char == "y":
		G.remove_edge(str((346,-103)),str((346,77)))
	# get rid of the self loops: nodes connected to itself
	G.remove_edges_from(G.selfloop_edges())

def find_path(list_of_degree):
	path = []
	if 1 in list_of_degree:
		node_index = list_of_degree.index(1)
		p = list(G.nodes)[node_index]
		path.append(eval(p))
		for i in range(G.number_of_edges()):
			# char R is a bit special...
			if char == "R" and p == str((163,352)):
				G.add_edge(str((163,352)),str((0,352)))
			if char == "a" and p == str((346,224)):
				G.add_edge(str((346,224)),str((346,77)))
			if char == "e" and p == str((0,77)):
				G.add_edge(str((0,77)),str((0,433)))
			if char == "m" and p == str((346,0)):
				G.add_edge(str((346,0)),str((346,433)))
			path,next_point = find_next_path(p,path)
			p = next_point
	else:
		p = find_top()
		path.append(eval(p))
		if char == "b":
			G.remove_edge(str((0,77)),str((0,433)))
		for i in range(G.number_of_edges()):
			# char B is a bit special...
			if char == "B" and p == str((163,352)):
				G.add_edge(str((163,352)),str((0,352)))
				print("added")

			path,next_point = find_next_path(p,path)
			p = next_point
	return path

# if the char only have one contour, use the contour directly
# otherwise put it in a graph for path planning
def main_alg(contours):
	n_contours = len(contours)
	# print(n_contours)
	outline_x = np.array([])
	outline_y = np.array([])
	if n_contours == 1 and char in only_contour_needed:
		outline_x,outline_y = path_for_easy_char(contours[0],outline_x,outline_y)
	else:
		graph_theory_init(contours)
		nodes_and_degree = list(G.degree)
		list_of_degree = [tup[1] for tup in nodes_and_degree]
		if draw_graph:
			nx.draw(G,with_labels=True)
			plt.show()
		path = find_path(list_of_degree)
		for i in range(len(path)):
			outline_x = np.append(outline_x,path[i][0])
			outline_y = np.append(outline_y,path[i][1])
	return outline_x,outline_y


outline_x,outline_y = main_alg(contours)
outline_total = np.array([outline_x, outline_y])
outline_total = outline_total.T

# np.savetxt("letters/{}.txt".format(ord(char)),outline_total,fmt=['%d','%d'])

	




# plotting
if not draw_graph:
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

		ani = FuncAnimation(fig, animate, frames=500, interval=100)
	else:
		plt.plot(outline_x,outline_y,marker='x')
		plt.gca().set_aspect('equal', adjustable='box')

plt.show()