import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

x_ori = np.array([[1,0,0]])
y_ori = np.array([[0,1,0]])
z_ori = np.array([[0,0,1]])
origin = np.array([[0,0,0]])
# np.random.seed(5)
sigma_r = 0.01
sigma = 0.5
phi = (np.random.rand()-0.5)*2*np.pi
theta = (np.random.rand()-0.5)*np.pi/8
number_of_measurements = 200
a = np.linspace(0,5,number_of_measurements)
b = np.random.normal(1,0.2,number_of_measurements)

def plane_rotation_matrix(phi,theta):
    R_z = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    R_x = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return np.dot(R_z,R_x)


def quiver_data_to_segments(X, Y, Z, u, v, w, length=1):
    segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
    segments = np.array(segments).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

def random_position():
    x = (np.random.rand())*10
    y = (np.random.rand())*10
    z = (np.random.rand())*10
    coord = np.array([[x,y,z]])
    return coord

def random_rotation_matrix():
    theta_z = (np.random.rand()-0.5)*np.pi/6 # only pi/12 rotation
    theta_y = (np.random.rand()-0.5)*np.pi/3
    theta_x = (np.random.rand()-0.5)*np.pi/3 
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
        point_true = ray_origin + 10*(np.random.rand()-0.5)*x_ray + 10*(np.random.rand()-0.5)*z_ray
        point_noise = point_true + np.random.normal(0,sigma,size=(1,3))
        measurements = np.vstack((measurements,point_true[0]))
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

C_WT = plane_rotation_matrix(phi,theta) 
x_p,y_p,z_p = plane_frame(C_WT,w_x,w_y,w_z)
plane_coord = random_position() # shape(1,3)

measurements = measurements_from_motion(plane_coord[0],np.transpose(x_p),np.transpose(z_p))


X,Y,Z = zip(origin[0],origin[0],origin[0],plane_coord[0],plane_coord[0],plane_coord[0])
U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])


Pr_0 = np.diag(np.array([sigma_r,sigma_r,sigma_r,sigma_r,sigma_r])**2)
Q = Pr_0
H = np.dot(np.array([[0,1,0]]),np.transpose(C_WT)) # shape(1,3)
# initial state

r_km1 = plane_coord + np.random.rand(1,3) # shape(1,3)
theta_noise = theta+np.random.rand()*0.05
phi_noise = phi+np.random.rand()*0.05
x_km1 = np.array([np.append(r_km1,[theta_noise,phi_noise])])
# initial covariance matrix
P_km1 = Pr_0
# print(C_WT)

angle_km1 = np.array([[theta_noise,phi_noise]])
Pa_0 = np.diag(np.array([sigma_r,sigma_r])**2)
Qa = Pa_0
Pa_km1 = Pa_0


# def animate(i):
#     global angle_km1,Pa_km1,poses,points
#     angle_k_km1 = angle_km1
#     Pa_k_km1 = Pa_km1 + Qa
#     C_WT = plane_rotation_matrix(angle_km1[0][0],angle_km1[0][1])
#     x_p_est,y_p_est,z_p_est = plane_frame(C_WT,w_x,w_y,w_z)
#     H = np.array([[np.sin(phi)*np.sin(theta)*plane_coord[0][0]-np.sin(theta)*np.cos(phi)*plane_coord[0][1]+np.cos(theta)*plane_coord[0][2],
#     -np.cos(theta)*np.cos(phi)*plane_coord[0][0]-np.cos(theta)*np.sin(phi)*plane_coord[0][1]]])
#     y = np.dot(np.array([[0,1,0]]),np.dot(np.transpose(C_WT),np.transpose(np.array([measurements[i]-plane_coord[0]]))))
#     print(y)
#     R = sigma**2*(C_WT[0][1]**2+C_WT[1][1]**2+C_WT[2][1]**2)
#     S = np.dot(H,Pa_k_km1).dot(np.transpose(H)) + R #(1,1)
#     K = np.dot(Pa_k_km1,np.transpose(H))/S
#     angle_km1 = angle_k_km1 + np.transpose(np.dot(K,y))
#     Pa_km1 = np.dot((np.identity(2)-np.dot(K,H)),Pa_k_km1)
#     X,Y,Z = zip(origin[0],origin[0],origin[0],plane_coord[0],plane_coord[0],plane_coord[0],plane_coord[0],plane_coord[0],plane_coord[0])
#     U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p_est)[0],np.transpose(y_p_est)[0],np.transpose(z_p_est)[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])
#     poses.remove()
#     poses = ax.quiver(X,Y,Z,U,V,W,length=1.,normalize=True,color='rgbrgb')
#     # points.remove()
#     points = ax.scatter(measurements[i][0],measurements[i][1],measurements[i][2],s=1)




def animate(i):
    global poses,points,r_km1,P_km1,x_km1
    x_k_km1 = x_km1
    P_k_km1 = P_km1 + Q
    # print("p shape is ", P_k_km1.shape)
    theta_est = x_k_km1[0][3]
    phi_est = x_k_km1[0][4]
    C_WT = plane_rotation_matrix(theta_est,phi_est) 
    # print(C_WT)
    x_p_est,y_p_est,z_p_est = plane_frame(C_WT,w_x,w_y,w_z)
    # print("y_p is ", y_p)
    # print("y_p estimate is ", y_p_est)
    H = np.array([[-np.cos(theta_est)*np.sin(phi_est),np.cos(theta_est)*np.cos(phi_est),np.sin(theta_est),
    np.sin(phi_est)*np.sin(theta_est)*x_k_km1[0][0]-np.sin(theta_est)*np.cos(phi_est)*x_k_km1[0][1]+np.cos(theta_est)*x_k_km1[0][2],
    -np.cos(theta_est)*np.cos(phi_est)*x_k_km1[0][0]-np.cos(theta_est)*np.sin(phi_est)*x_k_km1[0][1]]])
    y = np.dot(np.array([[0,1,0]]),np.dot(np.transpose(C_WT),np.transpose(np.array([measurements[i]-x_k_km1[0][:3]]))))
    # y = np.dot(np.array([[0,1,0]]),np.dot(np.transpose(C_WT),np.transpose(np.array([measurements[i]-r_km1[0]]))))
    print(y) 
    # print(abs(y))
    # print(measurements[i])
    # print(H)
    # print("y shape is ",y.shape)
    R = sigma_r
    S = np.dot(H,P_k_km1).dot(np.transpose(H)) + R #(1,1)
    # print("s shape is ",S.shape)
    K = np.dot(P_k_km1,np.transpose(H))/S #(5,1)
    # print(K)
    # print("k shape is ",K.shape)
    x_km1 = x_k_km1 + np.transpose(np.dot(K,y)) #(1,5)
    P_km1 = np.dot((np.identity(5) - np.dot(K,H)),P_k_km1)
    r_km1 = np.array([x_k_km1[0][:3]])
    X,Y,Z = zip(origin[0],origin[0],origin[0],r_km1[0],r_km1[0],r_km1[0],plane_coord[0],plane_coord[0],plane_coord[0])
    U,V,W = zip(x_ori[0],y_ori[0],z_ori[0],np.transpose(x_p_est)[0],np.transpose(y_p_est)[0],np.transpose(z_p_est)[0],np.transpose(x_p)[0],np.transpose(y_p)[0],np.transpose(z_p)[0])
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





