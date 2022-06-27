import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.linalg import bicgstab
from scipy.sparse import coo_matrix, diags
from scipy.io import loadmat


def F(f):
    return np.linalg.norm(f)


def loss(sensor_count):
    count_scale=int(2048/sensor_count)
    f=np.zeros((sensor_count,1))
    for i in range(sensor_count):
        string_id=str(1+i*count_scale)
        data_num=np.loadtxt('data'+str(i+1)+'.txt')
        data_orig=loadmat('data_'+string_id+'_emit.mat')['e']
        data_orig_mat=np.zeros((np.size(data_orig,0),sensor_count))
        for j in range(sensor_count):
            data_orig_mat[:,j]=data_orig[:,j*count_scale]
        f[i]=np.linalg.norm(data_num-data_orig_mat)**2
    return f


def Jacobian(Jacob_old,f_per,f_ini,pertur,iteration,n):
    n=np.size(pertur,0)
    pertur_reshape=np.reshape(torch.reciprocal(pertur).cpu().numpy(),(n*n,1))
    Jacob=np.outer(f_per-f_ini,np.atleast_2d(pertur_reshape))
    """
    if(iteration==0):
        n=np.size(pertur,0)
        pertur_reshape=np.reshape(torch.reciprocal(pertur).cpu().numpy(),(n*n,1))
        Jacob=np.outer(f_per-f_ini,np.atleast_2d(pertur_reshape))
    if(iteration>0):
        Jacob=np.outer(f_per-f_ini-Jacob_old@pertur.cpu().numpy(),np.atleast_2d(pertur.cpu().numpy()))
        Jacob=Jacob/(np.linalg.norm(pertur.cpu().numpy()))**2
        Jacob=Jacob+Jacob_old
    """
    return Jacob


def u0(u,sensors,sensor_id):
    u[sensors[sensor_id,1],sensors[sensor_id,0]]=1
    return u


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
    count_scale=int(2048/sensor_count)
    tau=4*1e-8
    tau_max=4*1e-8
    scale = int(tau_max/tau)
    h=radius*2.0/n
    T=3750
    sensors=np.ndarray((sensor_count,2),int)
    phi=np.linspace(0, 2*np.pi*(sensor_count-1)/sensor_count,sensor_count)

    c_p = torch.nn.functional.pad(input=c, pad=(1, 1, 1, 1), mode='constant', value=0)
    c2=torch.mul(c,c)
    c2_up=torch.mul(c_p[:-2,1:-1],c_p[:-2,1:-1])
    c2_down=torch.mul(c_p[2:,1:-1],c_p[2:,1:-1])
    c2_left=torch.mul(c_p[1:-1,:-2],c_p[1:-1,:-2])
    c2_right=torch.mul(c_p[1:-1,2:],c_p[1:-1,2:])

    for l in range(sensor_count):
        sensors[l,0]=find(n, radius, radius*np.cos(3*np.pi/2-phi[l]))
        sensors[l,1]=find(n, radius, radius*np.sin(3*np.pi/2-phi[l]))
    for i in range(sensor_count):
        u=torch.zeros((n,n)).to("cuda")
        u_n=torch.zeros((n,n)).to("cuda")
        u_nn=torch.zeros((n,n)).to("cuda")
        ticks=np.zeros((T,sensor_count))
        u_nn=u0(u_nn,sensors,i)
        u_n=u_nn
        string_id=str(1+i*count_scale)
        data_orig=loadmat('data_'+string_id+'_emit.mat')['e']
        signal_mat=np.zeros((np.size(data_orig,0),1))
        signal_mat=data_orig[:,i*count_scale]

        for k in range(scale*(T)):
            
            u_n[sensors[i,1],sensors[i,0]]=signal_mat[k//scale]

            u_p = torch.nn.functional.pad(input=u_n, pad=(1, 1, 1, 1), mode='constant', value=0)
            

            u_up=u_p[:-2,1:-1]
            u_down=u_p[2:,1:-1]
            u_left=u_p[1:-1,:-2]
            u_right=u_p[1:-1,2:]

            u=tau**2/(2.0*h**2)*(
                torch.mul(u_right,c2_right+c2)+
                torch.mul(u_up,c2_up+c2)+
                torch.mul(u_left,c2+c2_left)+
                torch.mul(u_down,c2+c2_down)
            )+2*u-torch.mul(u,
                            tau**2/(2.0*h**2)*(c2_right+c2_up+4*c2+c2_left+c2_down)
                        )-u_nn
            """
            #отрисовка волны в среде
            plt.clf()
            axis=np.linspace(-radius,radius,n)
    
            cp = plt.contourf(axis, axis, u.cpu().numpy())
            plt.colorbar(cp)
            plt.draw()
            plt.gcf().canvas.flush_events()
            """
            if(k//scale>=80 and k%scale==0):
                for l in range(np.size(sensors,0)):
                    ticks[k//scale-80,l]=u[sensors[l,1],sensors[l,0]]
            u_nn=u_n
            u_n=u
        string_='data'
        num_str=str(i+1)
        data_file=open(string_+num_str+'.txt', "w+")
        np.savetxt(data_file, ticks)
        data_file.close()


def lm_method(x_ini, sensor_count, radius):
    maxiteration=20
    eps=1e-4
    lambda_k=0.0001
    nu=1.0001
    n=int(np.sqrt(len(x_ini)))
    plt.ion()
    z=np.reshape(x_ini, (n, n))
    axis=np.linspace(-radius, radius, n)
    cp = plt.contourf(axis, axis, z)
    plt.colorbar(cp)
    for iterate in range(maxiteration):
        if(iterate>0):
            x_ini=x
        
        #идеи по очистке и экономии памяти:
        #изменить на float32 -> вроде сразу в 2 раза урежем объём
        #gc python -> сборщик мусора
        #automatic mixed presision  -> аналогично
        #half presision перекастовать в 16 бит

        c=torch.from_numpy(z).to("cuda")
        
        wave_solve(c, sensor_count, radius)
        f_ini=loss(sensor_count)
        #f_ini=np.zeros((sensor_count, 1))
        pertur=(1000.0*torch.rand(n, n)-500.0*torch.ones(n, n)).to("cuda")

        c_per=c+pertur
        wave_solve(c_per, sensor_count,radius)
        f_per=loss(sensor_count)
        #f_per=np.zeros((sensor_count, 1))

        if(iterate==0):
            J=np.zeros((sensor_count, n*n))
        J=coo_matrix(Jacobian(J, f_per, f_ini,pertur,iterate,n))

        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()

        data_J=J.data
        indices_J=np.vstack((J.row, J.col))
        i=torch.LongTensor(indices_J)
        v=torch.FloatTensor(data_J)
        shape=J.shape
        cuda_J=torch.sparse_coo_tensor(i, v, shape, device=torch.device('cuda'))

        indices_J_t=np.vstack((J.col, J.row))
        i_t=torch.LongTensor(indices_J_t)
        shape_t=[J.shape[1], J.shape[0]]
        cuda_J_t=torch.sparse_coo_tensor(i_t, v, shape_t, device=torch.device('cuda'))
        A=torch.sparse.mm(cuda_J_t,cuda_J)

        rows=A._indices()[0]
        columns=A._indices()[1]
        i_diag=(rows==columns)*rows
        data_diag=A._values()[i_diag[:]]
        diag_A=torch.sparse_coo_tensor(torch.vstack((i_diag, i_diag)), data_diag, A.shape)


        A=A+lambda_k*diag_A
        b=-J.T@f_ini
        A_data=A._values().cpu().numpy()
        A_rows=rows.cpu().numpy()
        A_columns=columns.cpu().numpy()
        A_cpu=coo_matrix((A_data, (A_rows, A_columns)), shape=A.shape)

        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()

        p=bicgstab(A_cpu, b)
        x=x_ini+p[0]
        if (F(f_per)>=F(f_ini)):
            lambda_k*=nu
        else:
            lambda_k/=nu
        plt.clf()
        z=np.reshape(x,(n,n)) #отрисовка распределения
        cp = plt.contourf(axis, axis, z)
        plt.colorbar(cp)
        plt.draw()
        plt.gcf().canvas.flush_events()

        if(np.linalg.norm(x-x_ini)<eps):
            break
        

        

n=50
sensor_count=16
d=0.2
radius=d/2.0
c_massive=1500.0*np.ones(n*n,dtype=np.float32)
axis=np.linspace(-radius,radius,n,dtype=np.float32)
for i in range(n):
    for j in range(n):
        if(np.sqrt(axis[i]**2+axis[j]**2)>=radius):
            c_massive[n*i+j]=0

c_massive=lm_method(c_massive, sensor_count, radius)
#plt.clf()
z=np.reshape(c_massive,(n,n))
cp = plt.contourf(axis, axis, z)
plt.colorbar(cp)
plt.draw()
plt.gcf().canvas.flush_events()
print("!!!!!!!!!!!!!!!!!!!!!!!!!!job is done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
a=input()