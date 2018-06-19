import numpy as np 
import matplotlib.pyplot as plt 
import scipy.fftpack as ft

def path(init,stepsize,n):
    path = np.zeros((2,n))
    path[:,0] = init
    for i in range(n-1):
        theta = np.random.uniform(0,2*np.pi)
        path[:,i+1] = path[:,i] + stepsize*np.array([np.cos(theta),np.sin(theta)])
    return path

## DCT
#Average sparcity for n=100

average_x = np.array([0,0,0],dtype=float)
average_y = np.array([0,0,0],dtype=float)

for i in range(500) : 
    generated_path = path(np.array([0,0]),1,100)
    transformed_x = ft.dct(generated_path[0])
    transformed_y = ft.dct(generated_path[1])
    average_x[0] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.05)
    average_x[1] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.01)
    average_x[2] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.005)
    average_y[0] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.05)
    average_y[1] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.01)
    average_y[2] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.005)
    
average_x /= 500
average_y /= 500

print("5%% x sparcity for n=100 is %.2f"%(average_x[0]/100))
print("1%% x sparcity for n=100 is %.2f"%(average_x[1]/100))
print("0.5%% x sparcity for n=100 is %.2f"%(average_x[2]/100))
print("5%% y sparcity for n=100 is %.2f"%(average_y[0]/100))
print("1%% y sparcity for n=100 is %.2f"%(average_y[1]/100))
print("0.5%% y sparcity for n=100 is %.2f"%(average_y[2]/100))

#Average sparcity for n=500

average_x = np.array([0,0,0],dtype=float)
average_y = np.array([0,0,0],dtype=float)

for i in range(500) : 
    generated_path = path(np.array([0,0]),1,500)
    transformed_x = ft.dct(generated_path[0])
    transformed_y = ft.dct(generated_path[1])
    average_x[0] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.05)
    average_x[1] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.01)
    average_x[2] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.005)
    average_y[0] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.05)
    average_y[1] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.01)
    average_y[2] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.005)

    
average_x /= 500
average_y /= 500

print("5%% x sparcity for n=500 is %.2f"%(average_x[0]/500))
print("1%% x sparcity for n=500 is %.2f"%(average_x[1]/500))
print("0.5%% x sparcity for n=500 is %.2f"%(average_x[2]/500))
print("5%% y sparcity for n=500 is %.2f"%(average_y[0]/500))
print("1%% y sparcity for n=500 is %.2f"%(average_y[1]/500))
print("0.5%% y sparcity for n=500 is %.2f"%(average_y[2]/500))

#Average sparcity for n=1000

average_x = np.array([0,0,0],dtype=float)
average_y = np.array([0,0,0],dtype=float)

for i in range(500) : 
    generated_path = path(np.array([0,0]),1,1000)
    transformed_x = ft.dct(generated_path[0])
    transformed_y = ft.dct(generated_path[1])
    average_x[0] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.05)
    average_x[1] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.01)
    average_x[2] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.005)
    average_y[0] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.05)
    average_y[1] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.01)
    average_y[2] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.005)
    
average_x /= 500
average_y /= 500

print("5%% x sparcity for n=1000 is %.2f"%(average_x[0]/1000))
print("1%% x sparcity for n=1000 is %.2f"%(average_x[1]/1000))
print("0.5%% x sparcity for n=1000 is %.2f"%(average_x[2]/1000))
print("5%% y sparcity for n=1000 is %.2f"%(average_y[0]/1000))
print("1%% y sparcity for n=1000 is %.2f"%(average_y[1]/1000))
print("0.5%% y sparcity for n=1000 is %.2f"%(average_y[2]/1000))

#Average sparcity for n=5000

average_x = np.array([0,0,0],dtype=float)
average_y = np.array([0,0,0],dtype=float)

for i in range(500) : 
    generated_path = path(np.array([0,0]),1,5000)
    transformed_x = ft.dct(generated_path[0])
    transformed_y = ft.dct(generated_path[1])
    value = np.abs(transformed_x)/np.abs(transformed_x[0])<0.05
    average_x[0] += np.sum(value)
    average_x[1] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.01)
    average_x[2] += np.sum(np.abs(transformed_x)/np.abs(transformed_x[0])<0.005)
    average_y[0] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.05)
    average_y[1] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.01)
    average_y[2] += np.sum(np.abs(transformed_y)/np.abs(transformed_y[0])<0.005)
    
average_x /= 500
average_y /= 500

print("5%% x sparcity for n=5000 is %.2f"%(average_x[0]/5000))
print("1%% x sparcity for n=5000 is %.2f"%(average_x[1]/5000))
print("0.5%% x sparcity for n=5000 is %.2f"%(average_x[2]/5000))
print("5%% y sparcity for n=5000 is %.2f"%(average_y[0]/5000))
print("1%% y sparcity for n=5000 is %.2f"%(average_y[1]/5000))
print("0.5%% y sparcity for n=5000 is %.2f"%(average_y[2]/5000))