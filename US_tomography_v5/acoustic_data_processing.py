from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import numpy as np
import torch
from tkinter import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_speed(scaling, radius):
    # отрисовка распределения
    # извлечение размера сетки на оси
    n_sq = len(scaling)
    n = int(np.sqrt(n_sq))

    axis = np.linspace(-radius, radius, n)
    sound_speed = scaling

    z = np.reshape(sound_speed, (n, n))
    plt.clf()
    cp = plt.contourf(axis, axis, z, cmap='gray', levels=300, vmin=1490, vmax=1530)
    plt.xlabel("x, m")
    plt.ylabel("y, m")
    plt.title("sound speed, m/s")
    plt.colorbar(cp)
    plt.show()

def F(f):
    # минимизируемый функционал
    return np.linalg.norm(f) ** 2

def read_from_file(fileName, n_emitter):
    elements_num = 2048
    time_points_num = 3750

    offset = int((elements_num / 4) * time_points_num * (n_emitter - 1) * 2)
    with open(fileName, mode='rb') as file:
        file.seek(offset, 0)
        file_content = file.read(time_points_num * elements_num // 4 * 2)
        test_data = np.frombuffer(file_content, dtype=np.int16)
        data = np.reshape(test_data, (time_points_num, elements_num // 4))
    return data

def read_data_for_emitter(path_experiment, n_emitter):
    
    fileName1 = 'decode_data_01.bin'
    fileName2 = 'decode_data_02.bin'
    fileName3 = 'decode_data_03.bin'
    fileName4 = 'decode_data_04.bin'

    data1 = read_from_file(path_experiment+fileName1, n_emitter)
    data2 = read_from_file(path_experiment+fileName2, n_emitter)
    data3 = read_from_file(path_experiment+fileName3, n_emitter)
    data4 = read_from_file(path_experiment+fileName4, n_emitter)

    # all_data -- матрица размера (3750, 2048)
    all_data = np.hstack((data1, data2, data3, data4))
    return all_data


def loss(sensor_amount):
    # функция потерь (ошибки) между
    # расчетными данными и экспериментами
    ratio = int(2048 / sensor_amount)
    f = np.zeros(sensor_amount*sensor_amount)
    for i in range(sensor_amount):
        # экспериментальные
        data_orig = read_data_for_emitter(1 + i * ratio)
        # сгенерированные данные
        data_num = np.reshape(np.fromfile('data' + str(i + 1) + '.bin'), (3750, sensor_amount))
        for j in range(sensor_amount):
            f[sensor_amount * i + j] = np.linalg.norm(data_num[:,j] - data_orig[:, j * ratio]/400) ** 2
    return f

def source(n, sensors, sensor_amount, sensor_id, time, fm):
    sensor_ratio = int(np.size(sensors, 0) / sensor_amount)
    s = torch.zeros((n, n)).to(device)
    s[sensors[sensor_id * sensor_ratio, 1], sensors[sensor_id * sensor_ratio, 0]] = (1 - 2 * (np.pi * fm * time) ** 2) * np.exp(-(np.pi * fm * time) ** 2)
    return s

def u0(u, sensors, sensor_amount, sensor_id, time):
    # граничное условие функции сигнала u
    # кусочная функция
    sensor_ratio = int(np.size(sensors, 0) / sensor_amount)
    t_end = 8
    if (time <= t_end):
        # значение амплитуды в вольтах
        u[sensors[sensor_id * sensor_ratio, 1], sensors[sensor_id * sensor_ratio, 0]] = 100.0
    return u

def find(n, radius, point):
    # поиск точки в массиве оси axis
    axis = np.linspace(-radius, radius, n)

    # номер ближайшего на сетке узла к точке
    index = int(0)
    found = axis[0]
    n = len(axis)
    for i in range(n):
        if (np.abs(axis[i] - point) < np.abs(found - point)):
            found = axis[i]
            index = i
    return index

def wave_solve(scaling, sensors, sensor_amount, radius, path_model):
    # решение волнового уравнения
    true_amount_emits = np.size(sensors, 0)
    sensors_ratio = int(true_amount_emits / sensor_amount)
    if (device.type == 'cuda:0'):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # размер сетки на оси
    n2 = np.size(scaling, 0)
    n = int(np.sqrt(n2))

    # распределение скорости до масштабирования
    c_cpu = scaling

    c_cpu = np.reshape(c_cpu, (n, n))
    c = torch.from_numpy(c_cpu).to(device)
    c2 = torch.mul(c, c)

    # 1 тик с частотой 25 МГц
    # tau = 1
    # коэффициент масштаба времени
    T_scale = 4 * 1e-8
    fm = 3 * 10e6
    # шаг сетки
    h = radius * 2.0 / n

    # количество тиков
    # первые 80 тиков никто не фиксирует сигнал
    T_listen = 3750
    T_end = 80 + T_listen

    for i in range(sensor_amount):
        ticks = torch.zeros((T_listen, sensor_amount)).to(device)
        u_nn = torch.zeros((n, n)).to(device)

        u_nn = u0(u_nn, sensors, sensor_amount, i, 0)

        u_n = u_nn
        ticks[0, :] = u_nn[sensors[::sensors_ratio, 1], sensors[::sensors_ratio, 0]]
        ticks[1, :] = u_n[sensors[::sensors_ratio, 1], sensors[::sensors_ratio, 0]]

        for k in range(2, T_end):

            u_p = torch.nn.functional.pad(input = u_n, pad = (1, 1, 1, 1), mode = 'constant', value = 0)

            u_up = u_p[:-2, 1:-1]
            u_down = u_p[2:, 1:-1]
            u_left = u_p[1:-1, :-2]
            u_right = u_p[1:-1, 2:]

            # шаг численного метода
            # наложение сдвинутых матриц u и c
            u = T_scale ** 2 / (h ** 2) * torch.mul(c2, u_up + u_left + u_down + u_right - 4 * u_n)\
                + 2 * u_n - u_nn + T_scale ** 2 * source(n, sensors, sensor_amount, i, T_scale * k, fm)

            """
            if (k % 250 == 0):
                plt.clf()
                cp = plt.contourf(axis, axis, u.cpu().numpy())
                plt.colorbar(cp)
                plt.show()
            """

            # расчетные диаграммы
            if(k >= 80):
                ticks[k - 80, :] = u[sensors[::sensors_ratio, 1], sensors[::sensors_ratio, 0]]
            u_nn = u_n
            u_n = u

        # запись диаграмм в файл
        ticks_np = ticks.cpu().numpy()
        num_str = str(i + 1)

        ticks_np.tofile(path_model + 'data' + num_str + '.bin')
    if (device.type == 'cuda:0'):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def lm_method(scaling_ini, sensor_amount, radius, path_experiment, path_model):

    # размер сетки на оси
    n_sq = len(scaling_ini)
    n = int(np.sqrt(n_sq))

    maxiter = 7

    true_amount_emits = 2048
    blocks = 8
    empty_emitters = 8
    # координаты сенсоров, номера узлов на сетке
    sensors = np.zeros((true_amount_emits, 2), int)

    all_sensors_amount = true_amount_emits + blocks * blocks
    sensors_in_real_block = int(all_sensors_amount / blocks)
    sensors_in_model_block = int(true_amount_emits / blocks)

    phi_ini = np.linspace(0, 2 * np.pi * (all_sensors_amount - 1) / all_sensors_amount, all_sensors_amount)
    phi = np.zeros(true_amount_emits)
    for i in range(1, blocks + 1):
        phi[((i - 1) * sensors_in_model_block) : (i - 1) * sensors_in_model_block + sensors_in_model_block] = phi_ini[empty_emitters - 1 + (i - 1) * sensors_in_real_block : empty_emitters - 1 + (i - 1) * sensors_in_real_block + sensors_in_model_block]

    for i in range(true_amount_emits):
        sensors[i, 0] = find(n, radius, radius * np.cos(-phi[i]))
        sensors[i, 1] = find(n, radius, radius * np.sin(-phi[i]))

    axis = np.linspace(-radius, radius, n)
    out_circle=[]
    for i in range(n):
        for j in range(n):
            if(np.sqrt(axis[i]**2+axis[j]**2)>=radius):
                out_circle.append(n * i + j)

    # критерий остановки F(f_per)<eps
    eps = 1.0
    # коэффициент демпфирования
    lambda_k = 0
    # множитель для изменения коэффициента демпфирования
    nu = 1.0001
    c0 = 1
    a = 10
    gamma = 0.25
    split_uniform = np.hstack((range(-2, -1+1), range(2, 10+1)))
    size_spl_uni = np.size(split_uniform, 0)
    for iter in range(maxiter):

        plot_speed(scaling_ini, radius)
        c_iter = c0 / (1 + iter) ** gamma

        pertur = 1.0 * np.random.choice(split_uniform, n * n, p = float(1 / size_spl_uni) * np.ones(size_spl_uni))

        scaling_per = scaling_ini - c_iter * pertur
        scaling_per[out_circle[:]]=0

        wave_solve(scaling_per, sensors, sensor_amount, radius, path_model)
        f_ini = loss(sensor_amount, path_experiment, path_model)

        scaling_per = scaling_ini + c_iter * pertur
        scaling_per[out_circle[:]] = 0

        wave_solve(scaling_per, sensors, sensor_amount, radius, path_model)
        f_per = loss(sensor_amount, path_experiment, path_model)

        # для масштабирования
        max_loss = 100.0

        f_ini = f_ini / max_loss
        f_per = f_per / max_loss
        
        u = (f_per - f_ini) / (2 * c_iter)
        v = np.reciprocal(pertur)
        Jacob_matr = np.outer(u, v)

        # ускорение подсчета диагонали через суммы Эйнштейна
        diag_data = np.einsum('ij,ji->i', Jacob_matr.T, Jacob_matr)
        Jacob = aslinearoperator(Jacob_matr)
        Jacob_T = aslinearoperator(Jacob_matr.T)
        fun_x = lambda X: Jacob_T @ Jacob @ X + lambda_k * np.multiply(diag_data, X)

        b = -Jacob_T @ f_ini

        total_operator = LinearOperator(shape = (n_sq, n_sq), matvec = fun_x)

        # поиск решения системы [J^T*J+lambda_k*diag(J^T*J)]p=b
        p = cg(total_operator, b, maxiter=2)

        # шаг метода
        if (not np.isnan(p[0][0])):
            a = a /(1+iter)

            scaling_ini = scaling_ini + a * np.multiply(np.reciprocal(diag_data), b)
            scaling_ini[out_circle[:]] = 0.0
            np.savetxt(path_model + "scaling" + str(iter+1) + ".txt", scaling_ini)


        if (F(f_per) >= F(f_ini)):
            lambda_k *= nu
        else:
            lambda_k /= nu

        if (F(f_per * max_loss) < eps):
            break

        if (device.type == 'cuda:0'):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return scaling_ini

def main_fun():

    path_experiment = experiment_tf.get()
    path_model = model_tf.get()

    # сетка области n*n
    n = 1000
    # количество сенсоров
    sensor_amount = 2
    # диаметр УЗИ-томографа
    d = 0.22
    radius = d / 2.0

    torch.set_default_dtype(torch.float64)

    scaling=1490*np.ones(n*n)

    # метод Левенберга -- Марквардта
    scaling = lm_method(scaling, sensor_amount, radius, path_experiment, path_model)
    np.savetxt(path_model + "scaling.txt", scaling)

window = Tk()
window.title('Characteristics Estimating System')
window.geometry('400x300')

frame = Frame(
   window,
   padx=10,
   pady=10
)
frame.pack(expand=True)

experiment_lb = Label(
   frame,
   text="Path to experiment data  ",
)
experiment_lb.grid(row=3, column=1) 

model_lb = Label(
   frame,
   text="Path to model data  "
)
model_lb.grid(row=4, column=1)

experiment_tf = Entry(
   frame,
)
experiment_tf.grid(row=3, column=2, pady=5)

model_tf = Entry(
   frame,
)
model_tf.grid(row=4, column=2, pady=5)

cal_btn = Button(
   frame,
   text='Start Processing',
   command=main_fun
)
cal_btn.grid(row=5, column=2)

window.mainloop()
