#/tmp/pycharm_project_88
from scipy.io import loadmat, savemat
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import bicgstab, cg, cgs
import matplotlib.pyplot as plt
import numpy as np
import torch

sound_speed_ini = 1500.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_speed(scaling, radius):
    # отрисовка распределения
    # извлечение размера сетки на оси
    n_sq = len(scaling)
    n = int(np.sqrt(n_sq))

    axis = np.linspace(-radius, radius, n)

    eye_vec = np.ones(n * n)

    # распределение скорости до масштабирования
    #sound_speed = sound_speed_ini * eye_vec
    sound_speed=scaling

    """    for i in range(n):
        for j in range(n):
            if (np.sqrt(axis[i] ** 2 + axis[j] ** 2) >= radius):
                # в области вне радиуса скорости нет
                sound_speed[n * i + j] = 0.0"""

    #sound_speed = np.multiply(sound_speed, np.maximum(np.zeros(n*n), scaling))

    z = np.reshape(sound_speed, (n, n))
    plt.clf()
    cp = plt.contourf(axis, axis, z, levels=50, cmap = plt.cm.jet)
    plt.xlabel("x, m")
    plt.ylabel("y, m")
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

def read_data_for_emitter(n_emitter):
    path = '/home/farronych/20190114-185300_5B_Agar/'

    fileName1 = 'decode_data_01.bin'
    fileName2 = 'decode_data_02.bin'
    fileName3 = 'decode_data_03.bin'
    fileName4 = 'decode_data_04.bin'

    data1 = read_from_file(path+fileName1, n_emitter)
    data2 = read_from_file(path+fileName2, n_emitter)
    data3 = read_from_file(path+fileName3, n_emitter)
    data4 = read_from_file(path+fileName4, n_emitter)

    # all_data -- матрица размера (3750, 2048)
    all_data = np.hstack((data1, data2, data3, data4))
    return all_data

def loss(sensor_amount):
    # функция потерь (ошибки) между
    # расчетными данными и экспериментами
    ratio = int(2048 / sensor_amount)
    f = np.zeros(sensor_amount)
    for i in range(sensor_amount):
        # экспериментальные
        data_orig = read_data_for_emitter(1 + i * ratio)
        # фрагмент данных (каждый sensor_count)
        data_orig_fragment = np.zeros((data_orig.shape[0], sensor_amount))
        # сгенерированные данные
        data_num = np.reshape(np.fromfile('data' + str(i + 1) + '.bin'), data_orig_fragment.shape)
        for j in range(sensor_amount):
            data_orig_fragment[:, j] = data_orig[:, j * ratio]
        f[i] = np.linalg.norm(data_num - data_orig_fragment) ** 2
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


def wave_solve(scaling, sensors, sensor_amount, radius):
    # решение волнового уравнения
    true_amount_emits = np.size(sensors, 0)
    sensors_ratio = int(true_amount_emits / sensor_amount)
    if (device.type == 'cuda:0'):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # размер сетки на оси
    n2 = np.size(scaling, 0)
    n = int(np.sqrt(n2))

    axis = np.linspace(-radius, radius, n)

    eye_vec = np.ones(n * n)

    # распределение скорости до масштабирования
    #c_cpu = sound_speed_ini * eye_vec
    c_cpu=scaling

    """
        for i in range(n):
        for j in range(n):
            if (axis[i] ** 2 + axis[j] ** 2 >= radius ** 2):
                # вне радиуса скорости нет
                c_cpu[n * i + j] = 0.0
    """

    #c_cpu = np.multiply(c_cpu, np.maximum(np.zeros(n * n), scaling))
    c_cpu = np.reshape(c_cpu, (n, n))
    c = torch.from_numpy(c_cpu).to(device)
    c2 = torch.mul(c, c)

    # 1 тик с частотой 25 МГц
    # tau = 1
    # коэффициент масштаба времени
    T_scale = 4 * 1e-8

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
            u = T_scale ** 2 / (h ** 2) * torch.mul(c2, u_up + u_left + u_down + u_right - 4 * u_n) + 2 * u_n - u_nn + T_scale ** 2 * source(n, sensors, sensor_amount, i, k, 1/k)
            """            if (k % 250 == 0):
                plt.clf()
                cp = plt.contourf(axis, axis, u.cpu().numpy(), levels=50)
                plt.colorbar(cp)
                plt.show()"""

            # расчетные диаграммы
            if(k >= 80):
                ticks[k - 80, :] = u[sensors[::sensors_ratio, 1], sensors[::sensors_ratio, 0]]
            u_nn = u_n
            u_n = u

        # запись диаграмм в файл
        ticks_np = ticks.cpu().numpy()
        num_str = str(i + 1)

        ticks_np.tofile('data' + num_str + '.bin')
        #savemat('_data_' + num_str + '_.mat', {"e": ticks_np, "label": "modeling"})
    if (device.type == 'cuda:0'):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def lm_method(scaling_ini, sensor_amount, radius):
    # метод Левенберга -- Марквардта

    maxiter = 10

    # размер сетки на оси
    n_sq = len(scaling_ini)
    n = int(np.sqrt(n_sq))

    # критерий остановки F(f_per)<eps
    eps = 1e-6

    # коэффициент демпфирования
    lambda_k = 10

    # множитель для изменения коэффициента демпфирования
    nu = 1.0001

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

    ones_vec = np.ones(n * n)

    c0 = 1
    a = 1000
    gamma = 0.25
    split_uniform = np.hstack((range(-40, -10), range(10, 40)))
    size_spl_uni = np.size(split_uniform, 0)
    for iter in range(maxiter):

        plot_speed(scaling_ini, radius)
        c_iter = c0 / (1 + iter) ** gamma

        pertur = 1.0 * np.random.choice(split_uniform, n * n, p = float(1 / size_spl_uni) * np.ones(size_spl_uni))
        #pertur = 1.0 * np.random.randint(low = 10, high = 40, size = n * n)
        #pertur = 2 * np.random.binomial(1, 0.5, n * n) - ones_vec

        scaling_per = scaling_ini - c_iter * pertur

        wave_solve(scaling_per, sensors, sensor_amount, radius)
        f_ini = loss(sensor_amount)

        scaling_per = scaling_ini + c_iter * pertur

        wave_solve(scaling_per, sensors, sensor_amount, radius)
        f_per = loss(sensor_amount)

        # для масштабирования
        max_loss = 1000.0

        f_ini = f_ini / max_loss
        f_per = f_per / max_loss
        
        u = (f_per - f_ini) / (2 * c_iter)
        v = np.reciprocal(pertur)
        Jacob_matr = np.outer(u, v)
        diag_data = np.einsum('ij,ji->i', Jacob_matr.T, Jacob_matr)  # ускорение подсчета диагонали через суммы Эйнштейна
        Jacob = aslinearoperator(Jacob_matr)
        Jacob_T = aslinearoperator(Jacob_matr.T)
        fun_x = lambda X: Jacob_T @ Jacob @ X + lambda_k * np.multiply(diag_data, X)

        b = -Jacob_T @ f_ini

        total_operator = LinearOperator(shape = (n_sq, n_sq), matvec = fun_x)
        # поиск решения системы [J^T*J+lambda_k*diag(J^T*J)]x=b
        #p = bicgstab(total_operator, b, maxiter = 2)
        #p = cgs(total_operator, b, maxiter=2)
        p = cg(total_operator, b, maxiter=2)

        # шаг метода
        if (not np.isnan(p[0][0])):
            if(p[0][0] * a < 50):
                a = a * 10
            else:
                a = a / 100
            scaling_ini = scaling_ini + a * p[0]
            np.savetxt("scaling.txt", scaling_ini)

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
"""
# сетка области n*n
n = 10
# количество сенсоров
sensor_amount = 128
# диаметр УЗИ-томографа
d = 0.2
radius = d / 2.0

torch.set_default_dtype(torch.float64)

scaling = sound_speed_ini * np.ones(n * n)



axis = np.linspace(-radius, radius, n)
for i in range(n):
    for j in range(n):
        if ((axis[i] - (0.001*np.random.uniform())) ** 2 + (
                axis[j] - (-radius/2.0+0.001*np.random.uniform())) ** 2 <= (0.025+0.01*np.random.uniform()) ** 2):
            # вне радиуса скорости нет
            scaling[n * i + j] = 1520.0 + np.random.randint(low = 10, high = 40)
        if (np.sqrt(axis[i] ** 2 + axis[j] ** 2) >= radius):
            scaling[n * i + j] = 0




#scaling=np.loadtxt('scaling.txt')
# метод Левенберга -- Марквардта
scaling = lm_method(scaling, sensor_amount, radius)
np.savetxt("scaling.txt", scaling)
"""