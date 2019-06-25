import numpy as np


def plane_func_2d(x):
    y = 2*x+3
    return y

sigma = np.random.rand()


x = np.linspace(0,15,10000)
y = plane_func_2d(x)
y_noise = y + np.random.normal(0,sigma,size=(y.size))
x_ = np.array([[x[0],1]])
for i in range(1,x.shape[0]):
    x_ = np.vstack((x_,[[x[i],1]]))
    
R_tm1 = sigma**2
b_tm1 = np.array([1.6, 3.5])


for i in range(x_.shape[0]):
    d_t = (x_[i])*R_tm1*np.transpose(x_[i])+0.1
    K_t = R_tm1*(x_[i])/d_t
    #print(b_tm1, K_t, x_[i], b_tm1)
    b_t = b_tm1+K_t.dot(y_noise[i]-x_[i].dot(b_tm1))
    R_t = R_tm1 - K_t.dot((x_[i]))*(R_tm1)
    b_tm1 = b_t
    R_tm1 = R_t

print(b_t)
