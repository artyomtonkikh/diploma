from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import bicgstab
from scipy.sparse import coo_matrix
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_speed(scaling,radius):
    #отрисовка распределения
    n_sq=np.size(scaling,0)
    n=int(np.sqrt(n_sq))                                 #извлечение размера сетки на оси
    axis=np.linspace(-radius, radius, n)                 #численная ось
    sound_speed=1468.0*np.ones(n*n)                      #начальное распределение скорости до масштабирования
    for i in range(n):
        for j in range(n):
            if(np.sqrt(axis[i]**2+axis[j]**2)>=radius):
                sound_speed[n*i+j]=0                     #в области вне радиуса скорости нет
    sound_speed=sound_speed*np.exp(scaling-np.ones(n*n)) #масштабирование
    plt.clf()
    z=np.reshape(sound_speed,(n,n)) 
    cp = plt.contourf(axis, axis, z)
    plt.colorbar(cp)
    plt.draw()
    plt.gcf().canvas.flush_events()

def F(f):
    #минимизируемый функционал
    return np.linalg.norm(f)


def loss(sensor_count):
    #функция потерь (ошибки) между 
    #расчетными данными и экспериментами
    count_scale=int(2048/sensor_count)
    f=np.zeros((sensor_count,1))
    for i in range(sensor_count):
        string_id=str(1+i*count_scale)                      #составление имени файла
        data_num=np.loadtxt('data'+str(i+1)+'.txt')
        data_orig=loadmat('data_'+string_id+'_emit.mat')['e']
        data_orig_mat=np.zeros((np.size(data_orig,0),sensor_count))
        for j in range(sensor_count):
            data_orig_mat[:,j]=data_orig[:,j*count_scale]
        f[i]=np.linalg.norm(data_num-data_orig_mat)**2
    return f


def Jacobian(f_per,f_ini,pertur):
    #построение якобиана по:
    #начальной вектор-функции, возмущенной и вектору возмущений
    #типа SPSA
    Jacob=np.outer(f_per-f_ini,np.atleast_2d(pertur))
    return Jacob


def u0(u,sensors,sensor_id,time):
    #граничное условие 
    #функции сигнала u в сенсоре с номером sensor_id
    t_end=1.0/3.0*1e-6                                      #до момента времени соответствующего частоте 3МГц
    if(time<=t_end):
        u[sensors[sensor_id,1],sensors[sensor_id,0]]=100    #значение амплитуды в вольтах
    return u


def find(n, radius, point):
    #поиск точки в массиве оси axis
    axis=np.linspace(-radius,radius,n)
    index=int(0)                                            #будет номером ближайшего на сетке узла к точке
    found=axis[0]
    n=len(axis)
    for i in range(n):
        if (np.abs(axis[i]-point)<np.abs(found-point)):
            found=axis[i]
            index=i
    return index


def wave_solve(scaling, sensor_count, radius):
    #решение волнового уравнения
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
    n_sq=np.size(scaling,0)
    n=int(np.sqrt(n_sq))                                #извлечение размера сетки на оси
    axis=np.linspace(-radius, radius, n)
    c_cpu=1468.0*np.ones(n*n)                           #начальное распределение используемое процессором
    for i in range(n):
        for j in range(n):
            if(np.sqrt(axis[i]**2+axis[j]**2)>=radius):
                c_cpu[n*i+j]=0.0
    
    c_cpu=c_cpu*np.exp(scaling-np.ones(n*n))
    c_cpu=np.reshape(c_cpu,(n,n))
    c=torch.from_numpy(c_cpu).to("cuda")                #перенос масштабированного распределения на GPU
    
    tau=4*1e-8                                          #шаг по времени не должен превышать
    tau_max=4*1e-8                                      #соответствующий 1 тику с частотой 25 МГц
    
    scale = int(tau_max/tau)                            #кратность текущего шага и максимально возможно
                                                        #для увеличения массива моментов времени
    
    h=radius*2.0/n                                      #шаг сетки
    
    T=3750                                              #количество тиков (специфика экспериментов)
    
    sensors=np.ndarray((sensor_count,2),int)                               #массив координат сенсоров
    phi=np.linspace(0, 2*np.pi*(sensor_count-1)/sensor_count,sensor_count) #массив углов

    c_p = torch.nn.functional.pad(input=c, pad=(1, 1, 1, 1), mode='constant', value=0)
    c2=torch.mul(c,c)                               #квадрат скоростей
    
    c2_up=torch.mul(c_p[:-2,1:-1],c_p[:-2,1:-1])    #в численной схеме
    c2_down=torch.mul(c_p[2:,1:-1],c_p[2:,1:-1])    #используются сдвинутые
    c2_left=torch.mul(c_p[1:-1,:-2],c_p[1:-1,:-2])  #массивы распределения скоростей
    c2_right=torch.mul(c_p[1:-1,2:],c_p[1:-1,2:])   #название после c2_ соответствует направлению сдвига 

    for l in range(sensor_count):
        sensors[l,0]=find(n, radius, radius*np.cos(3*np.pi/2-phi[l]))
        sensors[l,1]=find(n, radius, radius*np.sin(3*np.pi/2-phi[l]))
    for i in range(sensor_count):
        u=torch.zeros((n,n)).to("cuda")
        u_n=torch.zeros((n,n)).to("cuda")
        u_nn=torch.zeros((n,n)).to("cuda")
        ticks=np.zeros((T,sensor_count))
        u_nn=u0(u_nn,sensors,i,0)
        u_n=u_nn

        for k in range(2, scale*(T)):
            
            u_n=u0(u_n,sensors,i,k*tau)
            u_p = torch.nn.functional.pad(input=u_n, pad=(1, 1, 1, 1), mode='constant', value=0)
            
            u_up=u_p[:-2,1:-1]                      #аналонично сдвигам c2
            u_down=u_p[2:,1:-1]
            u_left=u_p[1:-1,:-2]
            u_right=u_p[1:-1,2:]

            #шаг численного метода
            u=tau**2/(2.0*h**2)*(
                torch.mul(u_right,c2_right+c2)+
                torch.mul(u_up,c2_up+c2)+
                torch.mul(u_left,c2+c2_left)+
                torch.mul(u_down,c2+c2_down)
            )+2*u-torch.mul(u,
                            tau**2/(2.0*h**2)*(c2_right+c2_up+4*c2+c2_left+c2_down)
                        )-u_nn
            
            
            #отрисовка волны в среде
            """
            plt.clf()
            axis=np.linspace(-radius,radius,n)
    
            cp = plt.contourf(axis, axis, u.cpu().numpy())
            plt.colorbar(cp)
            plt.draw()
            plt.gcf().canvas.flush_events()
            """
            if(k//scale>=80 and k%scale==0):
                for l in range(np.size(sensors,0)):
                    ticks[k//scale-80,l]=u[sensors[l,1],sensors[l,0]] #массив расчетных диаграмм
            u_nn=u_n
            u_n=u
        string_='data'                                  #запись в файл диаграмм сигнала
        num_str=str(i+1)
        data_file=open(string_+num_str+'.txt', "w+")
        np.savetxt(data_file, ticks)
        data_file.close()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()


def lm_method(scaling_ini, sensor_count, radius):
    #метод Левенберга -- Марквардта
    maxiteration=20                     #ограничение итераций
    n_sq=np.size(scaling_ini,0)
    n=int(np.sqrt(n_sq))
    eps=1e-4                            #критерий остановки если решение найдется быстрее
    lambda_k=0.0001                     #шаг метода Л -- М
    nu=1.0001                           #множитель для изменения шага метода
    n=int(np.sqrt(len(scaling_ini)))
    plt.ion()
    for iterate in range(maxiteration):
        if(iterate>0):
            scaling_ini=scaling
        
        #очистка и экономия памяти:
        #изменить на float32 -> сразу в 2 раза урежется объём
        #gc python -> сборщик мусора
        #automatic mixed presision  -> смешанное урезание на волю программы
        #half presision перекастовать в 16 бит

        wave_solve(scaling_ini, sensor_count, radius)           #получение
        f_ini=loss(sensor_count)                                #невозмущенных потерь
        
        pertur=4.0*np.random.rand(n*n)-2.0*np.ones(n*n)
        scaling_per=scaling_ini+pertur                          #возмущение начального scaling'а (типа SPSA)
        
        wave_solve(scaling_per, sensor_count,radius)            #возмущенные
        f_per=loss(sensor_count)                                #потери
        
        if(iterate==0):
            J=np.zeros((sensor_count, n*n))
        J=coo_matrix(Jacobian(f_per, f_ini, pertur))
        J_T=J.T
        diag_data=np.zeros(J.shape[1])
        for i in range(J.shape[1]):
            diag_data[i]=J_T.tocsr()[i,:].toarray()@J.tocsr()[:,i].toarray()
        diag_array=np.arange(J.shape[1])
        matrix_from_diag=coo_matrix((lambda_k*diag_data,(np.vstack((diag_array,diag_array)))), (J.shape[1],J.shape[1]))
        total_operator=aslinearoperator(J_T)@aslinearoperator(J)+aslinearoperator(matrix_from_diag)
        
        #total_operator для экономии памяти, так как J имеет размер [sensor_count, n^2]
        #после умножения J^t*J будет матрица [n^2, n^2]
        
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
        
        b=-J.T@f_ini                                #шаг
        p=bicgstab(total_operator, b, maxiter=10)   #метода
        scaling=scaling_ini+p[0]                    #Л -- М

        if (F(f_per)>=F(f_ini)):                    #в зависимости от ухудшения/улучшения результата шаг:
            lambda_k*=nu                            #увеличиваем
        else:                                       #или
            lambda_k/=nu                            #уменьшаем
        plot_speed(scaling,radius)                  #отрисовка
        if(np.linalg.norm(scaling-scaling_ini)<eps):
            break
    return scaling_ini

        

n=1000                                             #сетка области
sensor_count=2048                                  #количество учитываемых сенсоров
d=0.2                                            #диаметр области УЗИ-томографа
radius=d/2.0                                     #его радиус
#torch.set_default_dtype(torch.float32)           #урезаем формат данных для экономии
scaling=np.ones(n*n)                             #вектор масштаба, чтобы не оптимизировать по вектору с величинами порядка 1000
axis=np.linspace(-radius,radius,n)               #массив численной оси
scaling=lm_method(scaling, sensor_count, radius) #запуск метода Левенберга -- Марквардта
plot_speed(scaling,radius)                       #отрисовка распределения
print("!!!!!!!!!!!!!!!!!!!!!!!!!!job is done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
a=input()