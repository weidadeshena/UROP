import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

x_ori = np.array([[1,0,0]])
y_ori = np.array([[0,1,0]])
z_ori = np.array([[0,0,1]])
origin = np.array([[0,0,0]])
np.random.seed(5)
sigma_r = 0.01
sigma = 0.5
number_of_measurements = 200
a = np.linspace(0,5,number_of_measurements)
b = np.random.normal(1,0.2,number_of_measurements)

def quiver_data_to_segments(X, Y, Z, u, v, w, length=1):
    segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
    segments = np.array(segments).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

def random_basis():
    x = np.random.rand(3)
    x /= np.linalg.norm(x)
    y = np.random.rand(3)
    y -= y.dot(x)*x
    y /= np.linalg.norm(y)
    z = np.cross(x,y)
    return x,y,z

def random_position():
    x = (np.random.rand())*10
    y = (np.random.rand())*10
    z = (np.random.rand())*10
    coord = np.array([[x,y,z]])
    return coord

def random_rotation_matrix():
    theta_z = (np.random.rand()-0.5)*np.pi/6
    theta_y = (np.random.rand()-0.5)*np.pi/3
    theta_x = (np.random.rand()-0.5)*np.pi/3 # only pi/12 rotation
    R_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
    R_y = np.array([[np.cos(theta_y),0,np.sin(theta_y)],[0,1,0],[-np.sin(theta_y),0,np.cos(theta_y)]])
    R_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
    R_total = np.dot(np.dot(R_z,R_y),R_x)
    return theta_z,theta_y,theta_x,R_total

def plane_frame(R,w_x,w_y,w_z):
    w_x_p = np.dot(R,w_x)
    w_y_p = np.dot(R,w_y)
    w_z_p = np.dot(R,w_z)
    w_x_p /= np.linalg.norm(w_x_p)
    w_y_p /= np.linalg.norm(w_y_p)
    w_z_p /= np.linalg.norm(w_z_p)
    return w_x_p,w_y_p,w_z_p

def generate_measurements(ray_origin,x_ray,z_ray):
    measurements = np.array([ray_origin])
    for i in range(number_of_measurements):
        point_true = ray_origin + 10*np.random.rand()*x_ray + 10*np.random.rand()*z_ray
        point_noise = point_true + np.random.normal(0,sigma,size=(1,3))
        measurements = np.vstack((measurements,point_noise[0]))
    return measurements

def measurements_from_motion(ray_origin,x_ray,z_ray):
    measurements = np.array([ray_origin])
    for i in range(number_of_measurements):
        point_true = ray_origin + a[i]*x_ray + b[i]*z_ray
        point_noise = point_true + np.random.normal(0,sigma,size=(1,3))
        measurements = np.vstack((measurements,point_true[0]))
    return measurements
    

w_x = np.transpose(x_ori)
w_y = np.transpose(y_ori)
w_z = np.transpose(z_ori)

theta_z,theta_y,theta_x,C_WT = random_rotation_matrix()
x_p,y_p,z_p = plane_frame(C_WT,w_x,w_y,w_z)
plane_coord = random_position() # shape(1,3)

measurements = measurements_from_motion(plane_coord[0],np.transpose(x_p),np.transpose(z_p))


X,Y,Z = zip(origin[0],origin[0],origin[0],plane_coord[0],plane_coord[0],plane_coord[0])
U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])


P_0 = np.diag(np.array([sigma_r,sigma_r,sigma_r])**2)
Q = P_0
R = sigma**2*(C_WT[1][0]**2+C_WT[1][1]**2+C_WT[1][2]**2)
H = np.dot(np.array([[0,1,0]]),np.transpose(C_WT)) # shape(1,3)
# initial state
r_km1 = plane_coord + np.random.rand(1,3) # shape(1,3)
# initial covariance matrix
P_km1 = P_0
# print(C_WT)




def animate(i):
    global poses,points,r_km1,P_km1,x_p,y_p,z_p,x_ori,y_ori,z_ori
    r_k_km1 = r_km1
    P_k_km1 = P_km1 + Q
    # print("p shape is ", P_k_km1.shape)
    # measurement
    y = np.dot(H,(np.transpose(np.array([measurements[i]])-r_k_km1))) #- np.linalg.norm(measurements[i])
    print(abs(y))
    # print(measurements[i])
    # print(H)
    # print("y shape is ",y.shape)
    S = np.dot(H,P_k_km1).dot(np.transpose(H)) + R #(1,1)
    # print("s shape is ",S.shape)
    K = np.dot(P_k_km1,np.transpose(H))/S #(3,1)
    # print(K)
    # print("k shape is ",K.shape)
    r_km1 = r_k_km1 + np.transpose(np.dot(K,y)) #(1,3)
    P_km1 = np.dot((np.identity(3) - np.dot(K,H)),P_k_km1)
    # r_km1 = r_k_km1
    # P_km1 = P_k_km1
    # points.set_data(measurements[i][0],measurements[i][1])
    # points.set_3d_properties(measurements[i][2])
    X,Y,Z = zip(origin[0],origin[0],origin[0],r_km1[0],r_km1[0],r_km1[0],plane_coord[0],plane_coord[0],plane_coord[0])
    U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])
    # segments = quiver_data_to_segments(X,Y,Z,U,V,W)
    # poses.set_segments(segments)
    poses.remove()
    poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
    # points.remove()
    points = ax.scatter(measurements[i][0],measurements[i][1],measurements[i][2],s=1)

    


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim([0,12])
ax.set_ylim([0,12])
ax.set_zlim([0,12])
poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
points = ax.scatter(measurements[0][0],measurements[0][1],measurements[0][2],s=1)
# points = ax.plot(measurements[0][0],measurements[0][1], measurements[0][2], linestyle="", marker="o")
ani = animation.FuncAnimation(fig, animate, frames=number_of_measurements,
                              interval=1, blit=False)

plt.show()





