# find the plane position using Kalman filter
import numpy as np
import matplotlib.pyplot as plt

theta = np.pi/6
sigma_r = np.random.rand()*0.1
sigma_m = np.random.rand()*0.1
dt = 0.01

x_0 = [0,0,0,np.cos(theta),np.sin(theta),0]
x_km1 = x_0
w_r = np.random.normal(0,sigma_r,size=(1,6))[0]
P_0 = np.diag(np.square(w_r))
P_km1_km1 = P_0
Q = P_0
F = np.array([[1,0,0,10*dt,0,0],[0,1,0,0,10*dt,0],[0,0,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,0]])
H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
z = np.array([[0.,0.]])
point = np.array([[0.,0.]])
for i in np.arange(0,10,dt):
    point[0][0]+=np.cos(theta)*10*dt
    point[0][1]+=np.sin(theta)*dt*10
    x = point[0][0] + np.random.normal(0,sigma_m)
    y = point[0][1] + np.random.normal(0,sigma_m)
    z = np.append(z,[[x,y]],axis=0)
R = np.array([[sigma_m**2,0],[0,sigma_m**2]])
trajectory = point
k = 0
for i in np.arange(0,10,dt):
    #prediction
    x_k_km1 = F.dot(x_km1) 
    P_k_km1 = F.dot(P_km1_km1).dot(np.transpose(F))+Q
    #measurement
    # z is known
    y_k =np.transpose(z[k]) - np.matmul(H,x_k_km1)
    S_k = H.dot(P_k_km1).dot(np.transpose(H))+R
    K = P_k_km1.dot(np.transpose(H)).dot(np.linalg.inv(S_k))
    x_k_k = x_k_km1 + K.dot(y_k)
    P_k_k = (np.identity(6)-K.dot(H)).dot(P_k_km1)
    k += 1
    position = np.array([[x_k_k[0], x_k_k[1]]])
    trajectory = np.append(trajectory,position,axis=0)
    x_km1 = x_k_k
    P_km1_km1 = P_k_k

trajectory_plt = np.transpose(trajectory)
plt.plot(trajectory_plt[0], trajectory_plt[1])
plt.show()



