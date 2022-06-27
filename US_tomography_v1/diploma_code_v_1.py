import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse import csc_matrix, diags
import time
import torch

def vector_function(x, matrix):
    #data=4*1e-8*np.loadtxt('data.txt', dtype="float")
    data=100*np.ones(np.size(matrix,0))#.to("cuda")
    m=len(data)
    vector_data=np.zeros(m)##.to("cuda")
    fi=np.zeros(m)##.to("cuda")
    x_pseudo_inv=np.reciprocal(x)##.to("cuda")
    fi=matrix@x_pseudo_inv
    vector_data=fi-data
    return vector_data

def F(x, matrix):
    return (np.linalg.norm(vector_function(x, matrix)))**2

def method(x_initial, sensor_count, axis):
    maxiteration=20
    eps=1e-4
    lambda_k=0.0001
    nu=1.0001
    plt.ion()
    n=len(axis)
    z=np.reshape(x_initial, (n, n))##.to("cuda")
    cp = plt.contourf(axis, axis, 25.0*1e6*z)
    plt.colorbar(cp)
    for i in range(maxiteration):
        if(i>0):
            x_initial=x
            J_old=J
        if(i%5==0):
            matrix=raytracing(x_initial, sensor_count, axis)
            matrix=csc_matrix(matrix)
        f_initial=vector_function(x_initial, matrix)
        if(i==0):    
            J_old=np.zeros((len(f_initial), n))##.to("cuda")
        J=Jacob(x_initial, f_initial, matrix, J_old, i)
        J=csc_matrix(J)
        A=csc_matrix(J.T@J+lambda_k*diags((J.T@J).diagonal()))
        b=-J.T@f_initial
        p=bicgstab(A,b)
        x=x_initial+p[0]
        if (F(x, matrix)>=F(x_initial, matrix)):
            lambda_k*=nu
        else:
            lambda_k/=nu
        plt.clf()
        z=np.reshape(x,(n,n)) #отрисовка распределения
        cp = plt.contourf(axis, axis, 25.0*1e6*z)
        plt.colorbar(cp)
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.02)
        if(np.linalg.norm(x-x_initial)<eps):
            break
    
    return x


def Jacob(x, f, matrix, Jacobi_old, iteration):
    delta=5.0*1e-6/25.0
    m=len(f)
    n=len(x)
    Jacobian=np.zeros((m, n))#.to("cuda")
    if(iteration==0):
        perturbation_vector=np.zeros(n)#.to("cuda")
        f_perturbation=np.zeros(m)#.to("cuda")
        for j in range(n):
            perturbation_vector[j]=delta
            f_perturbation=vector_function(x+perturbation_vector, matrix)
            Jacobian[:,j]=(f_perturbation-f)/delta
            perturbation_vector[j]=0
    else:
        perturbation_vector=delta*np.ones(n)#.to("cuda")
        f_perturbation=vector_function(x+perturbation_vector, matrix)
        Jacobian=np.outer(f_perturbation-f-Jacobi_old@perturbation_vector,np.atleast_2d(perturbation_vector))
        Jacobian=Jacobian/(np.linalg.norm(perturbation_vector))**2
        Jacobian=Jacobian+Jacobi_old
    return Jacobian

def func(id_equation, massive_c, axis, y1, y2, x):
    #общая функция системы
    #id_equation - номер уравнения в системе, может быть 1 или 2
    #massive_c - массив скоростей звука
    #axis - оси (сетка равномерная)
    #y1 - искомый вектор в 1 уравнении
    #y2 - искомый вектор во 2 уравнении
    #x - точка в которой вычисляется значение функции
    #output - значение функции в точке x

    if(id_equation==1):
        func=func1(massive_c, axis, y1, y2, x)
    else:
        func=func2(massive_c, axis, y1, y2, x)
    return func


def func1(massive_c, axis, y1, y2, x):
    #функция 1 в системе
    #massive_c - массив скоростей звука
    #axis - оси (сетка равномерная)
    #y1 - искомый вектор в 1 уравнении
    #y2 - искомый вектор во 2 уравнении
    #x - точка в которой вычисляется значение функции
    #output - значение функции в точке x

    func1=np.zeros(2)#.to("cuda")
    n=len(axis)
    j=find(axis, x[0])
    i=find(axis, x[1])
    func1=massive_c[n*i+j]*y2/np.linalg.norm(y2)
    return func1


def func2(massive_c, axis, y1, y2, x):
    #функция 2 в системе
    #massive_c - массив скоростей звука
    #axis - оси (сетка равномерная)
    #y1 - искомый вектор в 1 уравнении
    #y2 - искомый вектор во 2 уравнении
    #x - точка в которой вычисляется значение функции
    #output - значение функции в точке x
    
    c0=1000.0*1e-6/25.0 #скорость звука в воде
    func2=np.zeros(2)#.to("cuda")
    n=len(axis)
    j=find(axis, x[0])
    i=find(axis, x[1])
    func2=-c0*gradient(massive_c, axis, y2, x)/massive_c[n*i+j]
    return func2


def gradient(massive_c, axis, y2, x):
    #градиент в точке x
    #massive_c - массив скоростей звука
    #axis - оси (сетка равномерная)
    #y2 - второй искомый вектор в системе
    #нужен для учёта направления при расчёте градиента
    #x - точка в которой вычисляется значение градиента
    #output - значение градиента в точке x

    h=axis[1]-axis[0]
    gradient_c=np.zeros(2)#.to("cuda")
    n=len(axis) #размер сетки
    j=find(axis, x[0])
    i=find(axis, x[1])
    sgn_x=int(np.sign(y2[0])) #учёт направления при расчёте по вектору y2
    sgn_y=int(np.sign(y2[1]))
    if(n*i+j+sgn_x*1>=n**2 or n*(i+sgn_y*1)+j>=n**2):
        #gradient_c=h-h
        h=h
    else:
        gradient_c[0]=(massive_c[n*i+j+sgn_x*1]-massive_c[n*i+j])/h
        gradient_c[1]=(massive_c[n*(i+sgn_y*1)+j]-massive_c[n*i+j])/h
    return gradient_c

def find(axis, point):
    #поиск ближайшего к точке узла сетки
    #axis - рассматриваемая ось
    #point - точка, которую требуется найти на оси axis
    #output - индекс ближайшей к точке point точки на оси axis 
    
    index=0
    found=axis[0]
    n=len(axis)
    for i in range(n):
        if (np.abs(axis[i]-point)<np.abs(found-point)):
            found=axis[i]
            index=i
    return index

def search(sensors, x, y):
    amount=np.size(sensors,0)
    sensor_to_search=np.zeros(2)#.to("cuda")
    sensor_to_search[0]=float(x)
    sensor_to_search[1]=float(y)
    sensor_index=0
    found=sensors[0,:]
    for k in range(amount):
        if(np.linalg.norm(found-sensor_to_search)>np.linalg.norm(sensors[k,:]-sensor_to_search)):
            found=sensors[k,:]
            sensor_index=k
    return sensor_index


def runge_kutta(delta_t, id_equation, massive_c, axis, y1_old, y2_old, x):
    #метод Рунге - Кутты 4 порядка
    #id_equation - номер уравнения в системе, может быть 1 или 2
    #massive_c - массив скоростей звука
    #axis - оси (сетка равномерная)
    #y1_old - значение искомого вектора в 1 уравнении на текущий момент
    #y2_old - значение искомого вектора в 2 уравнении на текущий момент
    #x - точка, в которой находимся в данный момент
    #output - новое значение функции
    id_equation=int(id_equation)
    h=axis[1]-axis[0]
    perturbation=np.ones(np.size(x,0))#.to("cuda")
    k1=func(id_equation, massive_c, axis, y1_old, y2_old, x)
    k2=func(id_equation, massive_c, axis, y1_old+k1*h/2.0, y2_old+k1*h/2.0, x+perturbation*h/2.0)
    k3=func(id_equation, massive_c, axis, y1_old+k2*h/2.0, y2_old+k2*h/2.0, x+perturbation*h/2.0)
    k4=func(id_equation, massive_c, axis, y1_old+k3*h, y2_old+k3*h, x+perturbation*h)
    if(id_equation==1):
        vector_old=y1_old
    else:
        vector_old=y2_old
    vector_new=vector_old+delta_t/6.0*(k1+2*k2+2*k3+k4)
    return vector_new


def one_emit_tracing(j, tracing_matrix, massive_c, sensors, axis):
    delta_t=250.0
    y1_new=np.zeros(2)#.to("cuda")
    y2_new=np.zeros(2)#.to("cuda")
    axis_size=np.size(axis,0)
    Radius=(axis[axis_size-1]-axis[0])/2
    c_size=np.size(massive_c,0)
    sensor_count=np.size(sensors,0)
    for k in range(sensor_count):
        if(int(j)!=k):
            tracing=np.zeros(c_size)#.to("cuda")
            y2_new=sensors[k,:]-sensors[j,:]
            y1_new=sensors[j,:]
            time_step=0
            while(True):
                time_step+=1
                y1_old=y1_new
                y2_old=y2_new
                tracing[axis_size*find(axis, y1_old[1])+find(axis, y1_old[0])]=delta_t*massive_c[axis_size*find(axis, y1_old[1])+find(axis, y1_old[0])]
                y2_new=runge_kutta(delta_t, 2, massive_c, axis, y1_old, y2_old, y1_old)
                y1_new=runge_kutta(delta_t, 1, massive_c, axis, y1_old, y2_new, y1_old)
                if(np.linalg.norm(y1_new)>=Radius and time_step>1):
                    sensor_x=axis[find(axis, y1_new[0])]
                    sensor_y=axis[find(axis, y1_new[1])]
                    sensor_index=search(sensors, sensor_x, sensor_y)
                    tracing_matrix[sensor_index, :]=tracing
                    break
    #return tracing_matrix

def raytracing(massive_c, sensor_count, axis):
    Radius=(axis[np.size(axis,0)-1]-axis[0])/float(2)
    c_size=np.size(massive_c,0)
    tracing_matrix=np.zeros((sensor_count**2, c_size))#.to("cuda")
    sensors=np.zeros((sensor_count,2))#.to("cuda")
    phi=np.linspace(0, 2*np.pi*(sensor_count-1)/sensor_count,sensor_count)#.to("cuda")
    for k in range(np.size(sensors,0)):
        sensors[k,0]=axis[find(axis, Radius*np.cos(3*np.pi/2-phi[k]))]
        sensors[k,1]=axis[find(axis, Radius*np.sin(3*np.pi/2-phi[k]))]


    #y1_new=np.zeros(2)#.to("cuda")
    #y2_new=np.zeros(2)#.to("cuda")
    for j in range(sensor_count):
        one_emit_tracing(j,tracing_matrix[j:j+sensor_count,:],massive_c, sensors, axis)
        #for k in range(sensor_count):
        #    if(j!=k):
        #        tracing=np.zeros(c_size)#.to("cuda")
        #        y2_new=sensors[k,:]-sensors[j,:]
        #        y1_new=sensors[j,:]
        #        time_step=0
        #        while(True):
        #            time_step+=1
        #            y1_old=y1_new
        #            y2_old=y2_new
        #            tracing[axis_size*find(axis, y1_old[1])+find(axis, y1_old[0])]=h
        #            y2_new=runge_kutta(delta_t, 2, massive_c, axis, y1_old, y2_old, y1_old)
        #            y1_new=runge_kutta(delta_t, 1, massive_c, axis, y1_old, y2_new, y1_old)
        #            if(np.linalg.norm(y1_new)>=Radius and time_step>1):
        #                sensor_x=axis[find(axis, y1_new[0])]
        #                sensor_y=axis[find(axis, y1_new[1])]
        #                sensor_index=search(sensors, sensor_x, sensor_y)
        #                tracing_matrix[sensor_count*j+sensor_index, :]=tracing
        #                break
    return tracing_matrix


n=1000     #количество узлов сетки
sensor_count=16
c_massive=1000.0*1e-6/25.0*np.ones(n*n)##.to("cuda")  #массив скоройстей звука
D=0.2    #диаметр УЗИ-прибора
Radius=D/2.0   #радиус УЗИ-прибора
axis=np.linspace(-Radius,Radius,n)#.to("cuda")
c_massive=method(c_massive, sensor_count, axis)
plt.clf()
z=np.reshape(c_massive,(n,n)) #отрисовка распределения
cp = plt.contourf(axis, axis, 25.0*1e6*z)
plt.colorbar(cp)
plt.draw()
plt.gcf().canvas.flush_events()
print("!!!!!!!!!!!!!!!!!!!!!!!!!!job is done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
a=input()


""""
for i in range(n):
    for j in range(n):
        #if (math.sqrt((axis[j]+0.025)**2+(axis[i]+0.025)**2)<=0.025):
            #c[n*i+j]=100.0
        #else:
        c[n*i+j]=1000.0

sensor_count=16

sensors=np.zeros((sensor_count,2))
#sensors=torch.zeros((sensor_count,2))

#if torch.cuda.is_available():
#    sensors = sensors.to('cuda')

phi=np.linspace(0,2*math.pi,sensor_count+1)
#phi=torch.linspace(0,2*math.pi,sensor_count+1)

#if torch.cuda.is_available():
#    sens_x = sens_x.to('cuda')

dimen=np.size(sensors,0)
#dimen=sensors.size(dim=0)

#s2=torch.cuda.Stream()
#with torch.cuda.stream(s2):
for k in range(dimen):
    sensors[k,0]=axis[find(axis, Radius*np.cos(3*math.pi/2-phi[k]))]
    sensors[k,1]=axis[find(axis, Radius*np.sin(3*math.pi/2-phi[k]))]

Y1=np.zeros(2)
Y2=np.zeros(2)
#Y1=torch.zeros(2)
#Y2=torch.zeros(2)

plt.ion()
fig = plt.figure(1)
#ax = fig.add_subplot(111)

z=np.reshape(c,(n,n)) #отрисовка распределения
#z=torch.reshape(c,(n,n))
#z_cpu=z.cpu()
#cp = plt.contourf(axis.numpy(), axis.numpy(), z_cpu.numpy())
cp = plt.contourf(axis, axis, z)
plt.colorbar(cp) 
now=time.time()-start
for j in range(dimen):
    for k in range(dimen):
        if(j!=k):
            del Y1
            del Y2
            Y2=sensors[k,:]-sensors[j,:]
            Y1=sensors[j,:]
            time_step=0
            while(True):
                time_step+=1
                if(time_step>1):
                    y1_old=Y1[time_step-1,:]
                    y2_old=Y2[time_step-1,:]
                else:
                    y1_old=Y1
                    y2_old=Y2
                y2_new=runge_kutta(2, c, axis, y1_old, y2_old, y1_old)
                y1_new=runge_kutta(1, c, axis, y1_old, y2_new, y1_old)

                if(norma(y1_new)>=Radius and time_step>1):
                    plt.clf()
                    z=np.reshape(c,(n,n)) #отрисовка распределения
                    cp = plt.contourf(axis, axis, z)
                    plt.colorbar(cp)
                    plt.draw()
                    plt.gcf().canvas.flush_events()
                    time.sleep(0.02)
                    break
                Y1=np.vstack((Y1, y1_new))
                Y2=np.vstack((Y2, y2_new))

z=np.reshape(c,(n,n)) #отрисовка распределения
cp = plt.contourf(axis, axis, z)
plt.colorbar(cp)
plt.ioff() 
plt.show()
"""