import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

x_ori = np.array([[1,0,0]])
y_ori = np.array([[0,1,0]])
z_ori = np.array([[0,0,1]])
origin = np.array([[0,0,0]])
np.random.seed(16)

# random initialisation (both +&-)
# flipped: 1,2,3,11,5,10,14,18,20,22,23,27,30,33,35,39,40,42,49,50
# works fine: 4,8,9,6,7,12,13,15,19,21,24,25,28,29,32,34,37,41,43,44,45,48
# different axis aligned: 16,17,26,31,36,46
# orientaion wrong: 38,47

# for anomolies
# initialisation both +: 
# flipped: 38,31
# not flipped: 
# initialisation both -: 
# flipped: 16,36
# not flipped: 47,17,26,46

sigma_r = 0.01
sigma_angle = 0.001
sigma_p_r = 0.001
dt = 0.05
psi = np.pi/8
phi = np.pi/16
# psi = (np.random.rand()-0.5)*np.pi
# phi = (np.random.rand()-0.5)*0.2
number_of_measurements = 200


def plane_rotation_matrix(phi,psi):
    R_z = np.array([[np.cos(psi),np.sin(psi),0],[-np.sin(psi),np.cos(psi),0],[0,0,1]])
    R_x = np.array([[1,0,0],[0,np.cos(phi),np.sin(phi)],[0,-np.sin(phi),np.cos(phi)]])
    return np.dot(R_z,R_x)

def random_position():
    x = (np.random.rand())*6
    y = (np.random.rand())*6
    z = (np.random.rand())*6
    coord = np.array([[x,y,z]])
    return coord

def plane_frame(C_WT,w_x,w_y,w_z):
    C_TW = np.transpose(C_WT)
    w_x_p = np.dot(C_TW,w_x)
    w_y_p = np.dot(C_TW,w_y)
    w_z_p = np.dot(C_TW,w_z)
    return w_x_p,w_y_p,w_z_p

def generate_measurements(ray_origin,x_ray,z_ray):
    measurements = np.array([ray_origin])
    for i in range(number_of_measurements):
        point_true = ray_origin + 2*(np.random.rand()-0.5)*x_ray + 2*(np.random.rand()-0.5)*z_ray
        point_noise = point_true + np.random.normal(size=(1,3))*sigma_r
        measurements = np.vstack((measurements,point_noise[0]))
    return measurements


w_x = np.transpose(x_ori)
w_y = np.transpose(y_ori)
w_z = np.transpose(z_ori)
C_WT = plane_rotation_matrix(phi,psi) 
x_p,y_p,z_p = plane_frame(C_WT,w_x,w_y,w_z)
w_r_p = np.array([[1,1,1]]) # shape(1,3)
measurements = generate_measurements(w_r_p[0],np.transpose(x_p),np.transpose(z_p))

Pr_0 = np.diag(np.array([0.1,0.1,0.1,0.1,0.1]))
Q = dt*np.diag(np.array([sigma_p_r,sigma_p_r,sigma_p_r,sigma_angle,sigma_angle])**2)

# initial state
r_km1 = w_r_p + np.array([[0.1,0.1,0.1]]) # shape(1,3)
phi_noise = phi+0.1
psi_noise = psi+0.1
x_km1 = np.array([np.append(r_km1,[phi_noise,psi_noise])])
# initial covariance matrix
P_km1 = Pr_0


def animate(i):
    global poses,points,r_km1,P_km1,x_km1
    x_k_km1 = x_km1
    P_k_km1 = P_km1 + Q
    phi_est = x_k_km1[0][3]
    psi_est = x_k_km1[0][4]
    # update
    C_WT = plane_rotation_matrix(phi_est,psi_est)
    x_p_est,y_p_est,z_p_est = plane_frame(C_WT,w_x,w_y,w_z)
    delta_r = measurements[i]-x_k_km1[0][:3]
    y = np.dot(np.array([[0,1,0]]),np.transpose(C_WT)).dot(np.transpose(np.array([delta_r])))
    # print(y)
    H_phi = np.sin(phi_est)*np.sin(psi_est)*delta_r[0]+np.sin(phi_est)*np.cos(psi_est)*delta_r[1]+np.sin(phi_est)*delta_r[2]
    H_psi = -np.cos(phi_est)*np.cos(psi_est)*delta_r[0]+np.cos(phi_est)*np.sin(psi_est)*delta_r[1]
    H_r = np.array([[0,1,0]]).dot(np.transpose(C_WT))
    H = np.array([np.append(H_r[0],[H_phi,H_psi])])
    # print(H)
    R = sigma_r**2*(C_WT[0][1]**2+C_WT[1][1]**2+C_WT[2][1]**2)
    S = np.dot(np.dot(H,P_k_km1),np.transpose(H)) + R
    K = np.dot(P_k_km1,np.transpose(H))/S
    # print("K: ",K)
    x_km1 = x_k_km1 + np.transpose(np.dot(K,y))
    P_km1 = np.dot((np.identity(5) - np.dot(K,H)),P_k_km1)
    r_km1 = np.array([x_k_km1[0][:3]])
    # print(np.dot(np.transpose(y_p),y_p_est))
    X,Y,Z = zip(origin[0],origin[0],origin[0],r_km1[0],r_km1[0],r_km1[0],w_r_p[0],w_r_p[0],w_r_p[0])
    U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p_est)[0],np.transpose(y_p_est)[0],
        np.transpose(z_p_est)[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])
    poses.remove()
    poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
    points.remove()
    points = ax.scatter(measurements[i][0],measurements[i][1],measurements[i][2],s=1)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
X,Y,Z = zip(origin[0],origin[0],origin[0],w_r_p[0],w_r_p[0],w_r_p[0])
U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])
poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
points = ax.scatter(measurements[0][0],measurements[0][1],measurements[0][2],s=1)
# points = ax.plot(measurements[0][0],measurements[0][1], measurements[0][2], linestyle="", marker="o")
ani = animation.FuncAnimation(fig, animate, frames=number_of_measurements,
                              interval=1, blit=False)

plt.show()


