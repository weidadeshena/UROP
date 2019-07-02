# test jacobian
import numpy as np

phi = (np.random.rand()-0.5)*2*np.pi
theta = (np.random.rand()-0.5)*np.pi/8
rx = np.random.rand()*10
ry = np.random.rand()*10
rz = np.random.rand()*10
delta = 0.01

h=-np.sin(phi)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi)*ry+np.sin(theta)*rz
# theta:
numerial_theta = (-np.sin(phi)*np.cos(theta+delta)*rx+np.cos(theta+delta)*np.cos(phi)*ry+np.sin(theta+delta)*rz - (-np.sin(phi)*np.cos(theta-delta)*rx+np.cos(theta-delta)*np.cos(phi)*ry+np.sin(theta-delta)*rz))/0.02
differential_theta = np.sin(phi)*np.sin(theta)*rx-np.sin(theta)*np.cos(phi)*ry+np.cos(theta)*rz
print("theta difference: ",numerial_theta-differential_theta)

# phi:
numerial_phi = (-np.sin(phi+delta)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi+delta)*ry+np.sin(theta)*rz - (-np.sin(phi-delta)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi-delta)*ry+np.sin(theta)*rz))/0.02
differential_phi = -np.cos(phi)*np.cos(theta)*rx-np.sin(phi)*np.cos(theta)*ry
print("phi difference: ",numerial_phi-differential_phi)

# rx:
numerial_rx = (-np.sin(phi)*np.cos(theta)*(rx+delta)+np.cos(theta)*np.cos(phi)*ry+np.sin(theta)*rz - (-np.sin(phi)*np.cos(theta)*(rx-delta)+np.cos(theta)*np.cos(phi)*ry+np.sin(theta)*rz))/0.02
differential_rx = -np.sin(phi)*np.cos(theta)
print("rx difference: ",numerial_rx-differential_rx)

# ry:
numerial_ry = (-np.sin(phi)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi)*(ry+delta)+np.sin(theta)*rz-(-np.sin(phi)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi)*(ry-delta)+np.sin(theta)*rz))/0.02
differential_ry = np.cos(theta)*np.cos(phi)
print("ry difference: ",numerial_ry-numerial_ry)

# rz:
numerial_rz = (-np.sin(phi)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi)*ry+np.sin(theta)*(rz+delta)-(-np.sin(phi)*np.cos(theta)*rx+np.cos(theta)*np.cos(phi)*ry+np.sin(theta)*(rz-delta)))/0.02
differential_rz = np.sin(theta)
print("rz difference: ",numerial_rz-differential_rz)
