# test jacobian
import numpy as np

# phi = (np.random.rand()-0.5)*2*np.pi
# theta = (np.random.rand()-0.5)*np.pi/8
phi = 0
theta = np.pi/2
rx = np.random.rand()*10
ry = np.random.rand()*10
rz = np.random.rand()*10
delta = 0.01
x_ori = np.array([[1,0,0]])
y_ori = np.array([[0,1,0]])
z_ori = np.array([[0,0,1]])
origin = np.array([[0,0,0]])
w_x = np.transpose(x_ori)
w_y = np.transpose(y_ori)
w_z = np.transpose(z_ori)

def plane_rotation_matrix(theta,phi):
    R_z = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    R_x = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return np.dot(R_z,R_x)

def plane_frame(R,w_x,w_y,w_z):
    w_x_p = np.dot(R,w_x)
    w_y_p = np.dot(R,w_y)
    w_z_p = np.dot(R,w_z)
    w_x_p /= np.linalg.norm(w_x_p)
    w_y_p /= np.linalg.norm(w_y_p)
    w_z_p /= np.linalg.norm(w_z_p)
    return w_x_p,w_y_p,w_z_p

C_WT = plane_rotation_matrix(theta,phi)
# print("C_WT: ")
# print(plane_rotation_matrix(theta,phi))
# C_WT = np.array([[np.cos(phi),-np.sin(phi)*np.cos(theta),np.sin(theta)*np.sin(phi)],[np.sin(phi),np.cos(phi)*np.cos(theta),-np.sin(theta)*np.cos(phi)],[0,np.sin(theta),np.cos(theta)]])
# print(C_WT)
# print(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))))
# print(np.array([-np.sin(phi)*np.cos(theta),np.cos(phi)*np.cos(theta),np.sin(theta)]))

x_p,y_p,z_p = plane_frame(C_WT,w_x,w_y,w_z)
print(x_p,y_p,z_p)

# h=np.dot(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))),np.array([rx,ry,rz]))
# print(h)
# k = -np.sin(phi)*np.cos(theta)*rx+np.cos(phi)*np.cos(theta)*ry+np.sin(theta)*rz
# print(k)
# # theta:
# numerial_theta = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta+delta,phi))).dot(np.array([rx,ry,rz]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta-delta,phi))).dot(np.array([rx,ry,rz]))))/0.02
# differential_theta = np.sin(phi)*np.sin(theta)*rx-np.sin(theta)*np.cos(phi)*ry+np.cos(theta)*rz
# print("theta: ",numerial_theta,differential_theta)

# # phi:
# numerial_phi = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi+delta))).dot(np.array([rx,ry,rz]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi-delta))).dot(np.array([rx,ry,rz]))))/0.02
# differential_phi = -np.cos(phi)*np.cos(theta)*rx-np.sin(phi)*np.cos(theta)*ry
# print("phi: ",numerial_phi,differential_phi)

# # rx:
# numerial_rx = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx+delta,ry,rz])) - (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx-delta,ry,rz]))))/0.02
# differential_rx = -np.sin(phi)*np.cos(theta)
# print("rx: ",numerial_rx,differential_rx)

# # ry:
# numerial_ry = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry+delta,rz]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry-delta,rz]))))/0.02
# differential_ry = np.cos(theta)*np.cos(phi)
# print("ry: ",numerial_ry,numerial_ry)

# # rz:
# numerial_rz = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry,rz+delta]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry,rz-delta]))))/0.02
# differential_rz = np.sin(theta)
# print("rz: ",numerial_rz,differential_rz)

X,Y,Z = zip(origin[0],origin[0],origin[0])
U,V,W = zip(x_p[0],y_p[0],z_p[0])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim([0,12])
ax.set_ylim([0,12])
ax.set_zlim([0,12])
ax.quiver(origin[0][1],Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
plt.show()


