# test jacobian
import numpy as np

phi = (np.random.rand()-0.5)*2*np.pi
theta = (np.random.rand()-0.5)*np.pi/8
rx = np.random.rand()*10
ry = np.random.rand()*10
rz = np.random.rand()*10
delta = 0.01

def plane_rotation_matrix(theta,phi):
    R_z = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    R_x = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return np.dot(R_z,R_x)

# print("C_WT: ")
# print(plane_rotation_matrix(theta,phi))
# C_WT = np.array([[np.cos(phi),-np.sin(phi)*np.cos(theta),np.sin(theta)*np.sin(phi)],[np.sin(phi),np.cos(phi)*np.cos(theta),-np.sin(theta)*np.cos(phi)],[0,np.sin(theta),np.cos(theta)]])
# print(C_WT)
# print(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))))
# print(np.array([-np.sin(phi)*np.cos(theta),np.cos(phi)*np.cos(theta),np.sin(theta)]))
h=np.dot(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))),np.array([rx,ry,rz]))
print(h)
k = -np.sin(phi)*np.cos(theta)*rx+np.cos(phi)*np.cos(theta)*ry+np.sin(theta)*rz
print(k)
# theta:
numerial_theta = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta+delta,phi))).dot(np.array([rx,ry,rz]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta-delta,phi))).dot(np.array([rx,ry,rz]))))/0.02
differential_theta = np.sin(phi)*np.sin(theta)*rx-np.sin(theta)*np.cos(phi)*ry+np.cos(theta)*rz
print("theta: ",numerial_theta,differential_theta)

# phi:
numerial_phi = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi+delta))).dot(np.array([rx,ry,rz]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi-delta))).dot(np.array([rx,ry,rz]))))/0.02
differential_phi = -np.cos(phi)*np.cos(theta)*rx-np.sin(phi)*np.cos(theta)*ry
print("phi: ",numerial_phi,differential_phi)

# rx:
numerial_rx = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx+delta,ry,rz])) - (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx-delta,ry,rz]))))/0.02
differential_rx = -np.sin(phi)*np.cos(theta)
print("rx: ",numerial_rx,differential_rx)

# ry:
numerial_ry = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry+delta,rz]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry-delta,rz]))))/0.02
differential_ry = np.cos(theta)*np.cos(phi)
print("ry: ",numerial_ry,numerial_ry)

# rz:
numerial_rz = (np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry,rz+delta]))-(np.dot(np.array([[0,1,0]]),np.transpose(plane_rotation_matrix(theta,phi))).dot(np.array([rx,ry,rz-delta]))))/0.02
differential_rz = np.sin(theta)
print("rz: ",numerial_rz,differential_rz)




