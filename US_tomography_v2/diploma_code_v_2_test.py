from pickletools import string1
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse import coo_matrix, diags
import torch


def list_of_zeros(u,sensors):
    list_=[]
    for i in range(np.size(sensors,0)):
        if(u[sensors[i,0],sensors[i,1]]==0):
            list_.append(i)
    return list_

def list_of_non_zeros(u,sensors):
    list_=[]
    for i in range(np.size(sensors,0)):
        if(u[sensors[i,0],sensors[i,1]]!=0):
            list_.append(i)
    return list_

def signal(k):
    if(k<=1):
        return 10.0
    else:
        return 0.0

def find(n, radius, point):
    axis=np.linspace(-radius,radius,n)
    index=int(0)
    found=axis[0]
    n=len(axis)
    for i in range(n):
        if (np.abs(axis[i]-point)<np.abs(found-point)):
            found=axis[i]
            index=i
    return index

def wave_solve(c, sensor_count, radius):
    n=np.size(c,0)
    tau=2*1e-8
    h=radius*2/n
    T=3750
    sensors=np.ndarray((sensor_count,2),int)
    phi=np.linspace(0, 2*np.pi*(sensor_count-1)/sensor_count,sensor_count)
    for l in range(sensor_count):
        sensors[l,0]=find(n, radius, radius*np.cos(3*np.pi/2-phi[l]))
        sensors[l,1]=find(n, radius, radius*np.sin(3*np.pi/2-phi[l]))
    for l in range(sensor_count):
        for k in range(T):
            u=torch.zeros((n,n)).to("cuda")
            u_n=torch.zeros((n,n)).to("cuda")
            u_nn=torch.zeros((n,n)).to("cuda")
            u_nn[sensors[l,1],sensors[l,0]]=signal(k)
            u_n=u_nn
            ticks=np.zeros((T,sensor_count))
            tick=0
            s=0
            #list_non0=list_of_zeros(u,sensors)
            while(len(list_of_zeros(u,sensors))!=0 and signal(k)!=0):
                s+=1
                if(s%2==0):
                    tick+=1
                u_p = torch.nn.functional.pad(input=u_n, pad=(1, 1, 1, 1), mode='constant', value=0)
                c_p = torch.nn.functional.pad(input=c, pad=(1, 1, 1, 1), mode='constant', value=0)

                u_up=u_p[:-2,1:-1]
                u_down=u_p[2:,1:-1]
                u_left=u_p[1:-1,:-2]
                u_right=u_p[1:-1,2:]

                c_up=c_p[:-2,1:-1]
                c_down=c_p[2:,1:-1]
                c_left=c_p[1:-1,:-2]
                c_right=c_p[1:-1,2:]

                u=tau**2/(2.0*h**2)*(
                    torch.mul(u_up,torch.mul(c_up,c_up)+torch.mul(c,c))+
                    torch.mul(u_right,torch.mul(c_right,c_right)+torch.mul(c,c))+
                    torch.mul(u_down,torch.mul(c,c)+torch.mul(c_down,c_down))+
                    torch.mul(u_left,torch.mul(c,c)+torch.mul(c_left,c_left))
                )+torch.mul(u,(
                                2.0-tau**2/(2.0*h**2)*(
                                    torch.mul(c_up,c_up)+
                                    torch.mul(c_right,c_right)+
                                    4*torch.mul(c,c)+
                                    torch.mul(c_down,c_down)+
                                    torch.mul(c_left,c_left)
                                    )
                                )
                            )-u_nn
                plt.clf()
                axis=np.linspace(-radius,radius,n)
                
                cp = plt.contourf(axis, axis, u.cpu().numpy())
                plt.colorbar(cp)
                plt.draw()
                plt.gcf().canvas.flush_events()
                list_non0=list_of_non_zeros(u,sensors)
                for i in range(len(list_non0)):
                    if(tick>=80):
                        ticks[tick-80,i]=u[sensors[list_non0[i],0],sensors[list_non0[i],1]]
                u_nn=u_n
                u_n=u
        string1='data'
        num_str=str(l)
        file=open(string1+num_str+'.txt')
        file.write(ticks)


def lm_method(x_ini, sensor_count, radius):
    maxiteration=20
    eps=1e-4
    lambda_k=0.0001
    nu=1.0001
    plt.ion()
    n=int(np.sqrt(len(x_ini)))
    z=np.reshape(x_ini, (n, n))
    cp = plt.contourf(axis, axis, z)
    plt.colorbar(cp)
    for iterate in range(maxiteration):
        #if(iterate>0):
            #x_ini=x
            #J_old=J
        c=torch.from_numpy(z).to("cuda")
        ticks=wave_solve(c, sensor_count, radius)
        #f_ini=loss(ticks)

n=1000
sensor_count=16
c_massive=2000.0*np.ones(n*n)
d=0.2
radius=d/2.0
axis=np.linspace(-radius,radius,n)
c_massive=lm_method(c_massive, sensor_count, radius)
#plt.clf()
z=np.reshape(c_massive,(n,n))
cp = plt.contourf(axis, axis, z)
plt.colorbar(cp)
plt.draw()
plt.gcf().canvas.flush_events()
print("!!!!!!!!!!!!!!!!!!!!!!!!!!job is done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
a=input()