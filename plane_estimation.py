# Stefan's code imitation
# it's the same algorithm why doesn't mine work :'(
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# parameters
N = 100
sigma_r = 0.01
sigma_q_r = 0.001
sigma_angle = 0.001
dt = 0.05


def plane_frame(C_WT,w_x,w_y,w_z):
    w_x_p = np.dot(C_WT,w_x)
    w_y_p = np.dot(C_WT,w_y)
    w_z_p = np.dot(C_WT,w_z)
    return w_x_p,w_y_p,w_z_p

def board_rotation(phi,psi):
	C_z = np.array([[np.cos(psi),np.sin(psi),0],[-np.sin(psi),np.cos(psi),0],[0,0,1]])
	C_x = np.array([[1,0,0],[0,np.cos(phi),np.sin(phi)],[0,-np.sin(phi),np.cos(phi)]])
	return np.dot(C_z,C_x)

# ground truth
x_axis = np.array([[1],[0],[0]])
y_axis = np.array([[0],[1],[0]])
z_axis = np.array([[0],[0],[1]])
origin = np.array([[0,0,0]])
w_r_x = np.random.rand()*10
w_r_y = np.random.rand()*10
w_r_z = np.random.rand()*10
w_r_true = np.array([[w_r_x,w_r_y,w_r_z]])
phi_true = np.random.uniform(-0.1,0.1)
psi_true = np.random.uniform(-np.pi,np.pi) 
x_true = np.array([[w_r_x,w_r_y,w_r_z,phi_true,psi_true]])
C_WT_true = board_rotation(phi_true,psi_true)
x_p_true,y_p_true,z_p_true = plane_frame(C_WT_true,x_axis,y_axis,z_axis)

# initial state and covariance
P = np.diag(np.array([0.1,0.1,0.1,0.1,0.1]))
x = np.transpose(x_true) + np.linalg.cholesky(P).dot(np.random.rand(5,1))

R = np.diag(np.array([sigma_r**2,sigma_r**2,sigma_r**2]))
Q = dt*np.diag(np.array([sigma_q_r**2,sigma_q_r**2,sigma_q_r**2,sigma_angle**2,sigma_angle**2]))
def animate(i):
	global points,poses,P,x
	# generate measurement
	r_tilde_w = np.transpose(w_r_true)+C_WT_true.dot(np.transpose(np.array([[np.random.uniform(-1,1),0,np.random.uniform(-1,1)]])))+np.linalg.cholesky(R).dot(np.random.rand(3,1))
	points = ax.scatter(r_tilde_w[0][0],r_tilde_w[1][0],r_tilde_w[2][0],s=1)
	# prediction
	P = P + Q
	#update
	phi_est = x[3][0]
	psi_est = x[4][0]
	C_WT = board_rotation(phi_est,psi_est)
	delta_r = r_tilde_w - x[0:3,:]
	# print(delta_r)
	H_phi = delta_r[2][0]*np.cos(phi_est)-delta_r[1][0]*np.cos(psi_est)*np.sin(phi_est)+delta_r[0][0]*np.sin(phi_est)*np.sin(psi_est)
	H_psi = -np.cos(phi_est)*(delta_r[0][0]*np.cos(psi_est)-delta_r[1][0]*np.sin(psi_est))
	H_r = np.array([[0,1,0]]).dot(np.transpose(C_WT))
	# print(H_phi,H_psi,H_r)
	H = np.array([np.append(H_r[0],[H_phi,H_psi])])
	# print(H)
	y = np.array([[0,1,0]]).dot(np.dot(np.transpose(C_WT),delta_r))
	S = np.dot(np.dot(H,P),np.transpose(H))+sigma_r**2*(C_WT[1][0]**2+C_WT[1][1]**2+C_WT[1][2]**2)
	K = np.dot(P,np.transpose(H))/S
	delta_x = np.dot(K,y)
	# print(H)
	s = np.transpose(np.dot(C_WT,np.array([[0],[1],[0]])))*delta_x[0:3,:]
	delta_x[0:3,:] = np.dot(np.dot(s,C_WT),np.array([[0],[1],[0]]))
	x = x + delta_x
	P = P - np.dot(np.dot(K,H),P)
	w_r_p_est = np.transpose(x[0:3,:])
	x_p_est,y_p_est,z_p_est = plane_frame(C_WT,x_axis,y_axis,z_axis)
	X,Y,Z = zip(origin[0],origin[0],origin[0],w_r_true[0],w_r_true[0],w_r_true[0],w_r_p_est[0],w_r_p_est[0],w_r_p_est[0])
	U,V,W = zip(np.transpose(x_axis)[0],np.transpose(y_axis)[0],np.transpose(z_axis)[0],np.transpose(x_p_true)[0],
		np.transpose(y_p_true)[0],np.transpose(z_p_true)[0],np.transpose(x_p_est)[0],np.transpose(y_p_est)[0],np.transpose(z_p_est)[0])
	poses.remove()
	poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
X,Y,Z = zip(origin[0],origin[0],origin[0],w_r_true[0],w_r_true[0],w_r_true[0])
U,V,W = zip(np.transpose(x_axis)[0],np.transpose(y_axis)[0],np.transpose(z_axis)[0],np.transpose(x_p_true)[0],
	np.transpose(y_p_true)[0],np.transpose(z_p_true)[0])
poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=1, blit=False)

plt.show()









