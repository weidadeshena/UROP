# robotic fonts yea boiiii
import numpy as np
from math import floor

def H(n=6,size): 
	d = 2.9/(n-5) #distance between each point, total length of trajectory is 2.1
	start = np.array([0.1,0.9])
	trajectory = np.array([start])
	while i < 2.9:
		while i <= 0.8:
			trajectory = np.vstack((trajectory,np.array([0.1,0.9-i])))
			i+=d
		trajectory = np.vstack((trajectory,np.array([0.1,0.1])))
		while i > 0.8 and i <= 1.1:
			trajectory = np.vstack((trajectory,np.array([0.1,0.1+i-0.8])))
			i+=d
		trajectory = np.vstack((trajectory,np.array([0.1,0.6])))
		while i > 1.1 and i <= 1.8:
			trajectory = np.vstack((trajectory,np.array([i-1.7,0.6])))
			i+=d
		trajectory = np.vstack((trajectory,np.array([0.8,0.5])))
		while i>1.8 and i<=2.1:
			trajectory = np.vstack((trajectory,np.array([0.8,0.5+i-1.8])))
			i+=d
		trajectory = np.vstack((trajectory,np.array([0.8,0.9])))
		while i>2.1 and i <=2.9:
			trajectory = np.vstack((trajectory,np.array([0.8,0.9-i+2.1])))
			i+=d
		trajectory = np.vstack((trajectory,np.array([0.8,0.1])))


