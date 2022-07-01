from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_speed(scaling,radius):
    #отрисовка распределения

    n_sq=np.size(scaling, 0)
    #извлечение размера сетки на оси
    n=int(np.sqrt(n_sq))

    #численная ось
    axis=np.linspace(-radius, radius, n)

    #начальное распределение скорости до масштабирования
    sound_speed=1400.0*np.ones(n*n)
    for i in range(n):
        for j in range(n):
            if(np.sqrt(axis[i]**2+axis[j]**2)>=radius):
                #в области вне радиуса скорости нет
                sound_speed[n*i+j]=0
    
    #масштабирование
    sound_speed=np.multiply(sound_speed,np.exp(scaling-np.ones(n*n)))
    plt.clf()
    z=np.reshape(sound_speed, (n, n)) 
    cp = plt.contourf(axis, axis, z)
    plt.xlabel("x, m")
    plt.ylabel("y, m")
    plt.colorbar(cp, label="sound speed, m/s")
    plt.draw()
    plt.gcf().canvas.flush_events()

def F(f):
    #минимизируемый функционал
    #квадрат нормы вектора
    return torch.linalg.norm(f)**2


def loss(sensor_count):
    #функция потерь (ошибки) между 
    #расчетными данными и экспериментами
    count_scale=int(2048/sensor_count)
    f=np.zeros(sensor_count)
    for i in range(sensor_count):
        #составление имени файла
        string_id=str(1+i*count_scale)
        data_num=np.loadtxt('data'+str(i+1)+'.txt')
        data_orig=loadmat('data_'+string_id+'_emit.mat')['e']
        data_orig_mat=np.zeros((np.size(data_orig, 0), sensor_count))
        for j in range(sensor_count):
            data_orig_mat[:,j]=data_orig[:,j*count_scale]
        f[i]=np.linalg.norm(data_num-data_orig_mat)**2
    return f


def u0(u,sensors,sensor_id,time):
    #граничное условие 
    #функции сигнала u в сенсоре с номером sensor_id
    #до момента времени соответствующего частоте 3МГц
    t_end=1.0/3.0*1e-6
    if(time<=t_end):
        #значение амплитуды в вольтах
        u[sensors[sensor_id,1],sensors[sensor_id,0]]=100
    return u


def find(n, radius, point):
    #поиск точки в массиве оси axis
    axis=np.linspace(-radius, radius, n)

    #номер ближайшего на сетке узла к точке
    index=int(0)

    found=axis[0]
    n=len(axis)
    for i in range(n):
        if (np.abs(axis[i]-point)<np.abs(found-point)):
            found=axis[i]
            index=i
    return index

def SPSA(u, v, b, lambda_k):
    #решение системы уравнений [J^T*J+lambda_kdiag(J^T*J)]x=b
    #через решение оптимизационной задачи
    #||[J^T*J+lambda_kdiag(J^T*J)]x-b||^2->min

    #Якобиан через диадное произведение
    #J=(f_per-f_ini)⊗reciprocal(pertur) = (f_per-f_ini)*reciprocal(pertur)^T
    #reciprocal(pertur)_i=1/pertur_i
    #обозначив u=f_per-f_ini и v=reciprocal(pertur)
    
    n=b.shape
    with torch.no_grad():
        maxiter=10

        #коэффициенты в SPSA
        A=10
        c1=0.05
        a1=0.002

        #начальное решение
        x=torch.zeros(n).to('cuda')

        for iter in range(maxiter):
            a_iter=a1/((iter+1+A)**(0.602))
            c_iter=c1/((iter+1)**(0.101))

            #вектор возмущений из Радемахеровских случайных величин {-1,1}
            pertur=torch.rand(n)
            pertur=2.0*torch.bernoulli(pertur)-torch.ones(n)
            pertur=pertur.to('cuda')
            pertur=pertur.type(torch.float64)

            #возмущение начального решения назад
            x_per=x-c_iter*pertur

            #J_T*J*x
            y1=torch.dot(x_per,v)*torch.dot(u,u)*v

            #lambda_k*diag(J_T*J)*x
            y2=lambda_k*torch.dot(u,u)*torch.mul(v,v)*x_per

            #||(J_T*J+lambda_k*diag(J_T*J))*x-b||^2
            F_ini=F(y1+y2-b)

            #возмущение начального решения вперед
            x_per=x+c_iter*pertur

            #J_T*J*x
            y1=torch.dot(x_per,v)*torch.dot(u,u)*v

            #lambda_k*diag(J_T*J)*x
            y2=lambda_k*torch.dot(u,u)*torch.mul(v,v)*x_per

            #||(J_T*J+lambda_k*diag(J_T*J))*x-b||^2
            F_per=F(y1+y2-b)
            
            #∇F(x)=∇||(J_T*J+lambda_k*diag(J_T*J))*x-b||^2
            gradient=(F_per-F_ini)*torch.reciprocal(2*c_iter*pertur)

            x=x-a_iter*gradient
        return x

def wave_solve(scaling, sensor_count, radius):
    #решение волнового уравнения
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
    n_sq=np.size(scaling,0)
    n=int(np.sqrt(n_sq))
    axis=np.linspace(-radius, radius, n)

    #начальное распределение используемое процессором
    c_cpu=1400.0*np.ones(n*n)

    for i in range(n):
        for j in range(n):
            if(np.sqrt(axis[i]**2+axis[j]**2)>=radius):
                c_cpu[n*i+j]=0.0
    
    c_cpu=np.multiply(c_cpu,np.exp(scaling-np.ones(n*n)))
    c_cpu=np.reshape(c_cpu, (n, n))

    #перенос масштабированного распределения на GPU
    c=torch.from_numpy(c_cpu).to("cuda")
    
    #шаг по времени не должен превышать tau_max
    #tau_max соответствует 1 тику с частотой 25 МГц
    tau=4*1e-8
    tau_max=4*1e-8
    
    #кратность текущего шага и максимально возможного
    #для увеличения массива моментов времени
    scale = int(tau_max/tau)
    
    #шаг сетки
    h=radius*2.0/n
    
    #количество тиков (специфика экспериментов)
    T=3750
    
    #массив координат сенсоров
    sensors=np.ndarray((sensor_count, 2), int)

    #массив углов
    phi=np.linspace(0, 2*np.pi*(sensor_count-1)/sensor_count, sensor_count)

    c_p = torch.nn.functional.pad(input=c, pad=(1, 1, 1, 1), mode='constant', value=0)

    #квадрат скоростей
    c2=torch.mul(c, c)                               
    
    #в численной схеме используются сдвинутые
    #массивы распределения скоростей
    #название после c2_ соответствует направлению сдвига
    c2_up=torch.mul(c_p[:-2,1:-1], c_p[:-2,1:-1])
    c2_down=torch.mul(c_p[2:,1:-1], c_p[2:,1:-1])
    c2_left=torch.mul(c_p[1:-1,:-2], c_p[1:-1,:-2])
    c2_right=torch.mul(c_p[1:-1,2:], c_p[1:-1,2:])

    for i in range(sensor_count):
        sensors[i,0]=find(n, radius, radius*np.cos(3*np.pi/2-phi[i]))
        sensors[i,1]=find(n, radius, radius*np.sin(3*np.pi/2-phi[i]))
    for i in range(sensor_count):
        u=torch.zeros((n,n)).to("cuda")
        u_n=torch.zeros((n,n)).to("cuda")
        u_nn=torch.zeros((n,n)).to("cuda")
        ticks=np.zeros((T, sensor_count))
        u_nn=u0(u_nn, sensors, i, 0)
        u_n=u_nn

        for k in range(2, scale*(T)):
            
            u_n=u0(u_n, sensors, i, k*tau)
            u_p = torch.nn.functional.pad(input=u_n, pad=(1, 1, 1, 1), mode='constant', value=0)
            
            #аналонично сдвигам c2
            u_up=u_p[:-2,1:-1]
            u_down=u_p[2:,1:-1]
            u_left=u_p[1:-1,:-2]
            u_right=u_p[1:-1,2:]

            #шаг численного метода
            u=tau**2/(2.0*h**2)*(
                torch.mul(u_right, c2_right+c2)+
                torch.mul(u_up, c2_up+c2)+
                torch.mul(u_left, c2+c2_left)+
                torch.mul(u_down, c2+c2_down)
            )+2*u-torch.mul(u,
                            tau**2/(2.0*h**2)*(c2_right+c2_up+4*c2+c2_left+c2_down)
                        )-u_nn
            
            if(k//scale>=80 and k%scale==0):
                for l in range(np.size(sensors,0)):
                    #массив расчетных диаграмм
                    ticks[k//scale-80,l]=u[sensors[l,1],sensors[l,0]]
            u_nn=u_n
            u_n=u
        #запись диаграмм в файл
        string_='data'
        num_str=str(i+1)
        data_file=open(string_+num_str+'.txt', "w+")
        np.savetxt(data_file, ticks)
        data_file.close()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()


def lm_method(scaling_ini, sensor_count, radius):
    #метод Левенберга -- Марквардта

    #ограничение итераций
    maxiter=10
    n_sq=np.size(scaling_ini,0)
    n=int(np.sqrt(n_sq))

    #критерий остановки ||scaling_ini-scaling_new||<eps 
    #если решение найдется быстрее
    eps=1e-4

    #коэффициент демпфирования
    lambda_k=0.0001

    #множитель для изменения коэффициента демпфирования
    nu=1.0001
    n=int(np.sqrt(len(scaling_ini)))
    plt.ion()

    #коэффициент в SPSA
    c1=0.05

    for iter in range(maxiter):
        
        c_iter=c1/((iter+1)**(0.101))
        #вектор возмущений из Радемахеровских случайных величин {-1,1}
        #SPSA
        pertur=2.0*np.random.binomial(1,0.5,n*n)-np.ones(n*n)
        pertur=pertur.astype(dtype=np.float64)

        #возмущение начального scaling'а
        scaling_per=scaling_ini-c_iter*pertur

        #получение 1 возмущенных значений
        wave_solve(scaling_per, sensor_count, radius)
        f_ini=loss(sensor_count)

        #возмущение начального scaling'а
        scaling_per=scaling_ini+c_iter*pertur
        
        #получение 2 возмущенных значений
        wave_solve(scaling_per, sensor_count,radius)
        f_per=loss(sensor_count)
        
        #максимальный элемент f_per для масштабирования
        max_loss=np.amax(f_per)
        
        #масштабирование ошибок
        f_ini=f_ini/max_loss
        f_per=f_per/max_loss

        #Якобиан представляем через диадное произведение
        #J=(f_per-f_ini)⊗reciprocal(pertur) = (f_per-f_ini)*reciprocal(pertur)^T
        #reciprocal(pertur)_i=1/pertur_i
        #обозначим u=f_per-f_ini и v=reciprocal(pertur)
        u=torch.from_numpy(f_per-f_ini).to('cuda')
        v=torch.from_numpy(np.reciprocal(pertur)).to('cuda')

        #действие диадного произведения на вектор
        #(u⊗v)x=(x*v)*u
        
        #b=-J^T*f_ini
        #через диады:
        #b=-[(u⊗v)^T]f_ini=-(v⊗u)f_ini=-(f_ini*u)*v
        b=torch.dot(u,torch.from_numpy(-f_ini).to('cuda'))*v
        
        #решение оптимизационной задачи
        #||[J^T*J+lambda_kdiag(J^T*J)]x-b||^2->min
        #таким образом найдем решение системы
        p=SPSA(u, v, b, lambda_k)
        
        #шаг метода Л -- М
        p_cpu=p.cpu().numpy()

        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
        
        scaling_ini=scaling_ini+p_cpu

        #в зависимости от ухудшения/улучшения результата шаг:
        if (F(torch.from_numpy(f_per))>=F(torch.from_numpy(f_ini))):
            #увеличиваем
            lambda_k*=nu
        else:
            #или уменьшаем
            lambda_k/=nu
        
        #отрисовка
        plot_speed(scaling_ini, radius)
        
        if(F(torch.from_numpy(f_per))<eps):
            break
    return scaling_ini

        

#сетка области n*n
n=100

#количество учитываемых сенсоров
sensor_count=16

#диаметр области УЗИ-томографа
d=0.2
#его радиус
radius=d/2.0

torch.set_default_dtype(torch.float64) 

#вектор масштаба, чтобы не оптимизировать по вектору с величинами порядка 1000
scaling=np.ones(n*n)

#массив численной оси
axis=np.linspace(-radius, radius, n)

#запуск метода Левенберга -- Марквардта
scaling=lm_method(scaling, sensor_count, radius)

#отрисовка распределения
plot_speed(scaling, radius)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!job is done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
a=input()