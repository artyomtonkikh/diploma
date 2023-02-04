from unittest import TestCase, main
import numpy as np
import torch
from US_version4_3_for_test import find, source, read_from_file, u0, wave_solve

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class my_tests(TestCase):
    
    def test_find(self):
        n = 10
        radius = 5.0
        point = 0.0
        self.assertEqual(find(n, radius, point), 4)

    def test_source1(self):
        n = 100
        radius = 2.0
        sensor_amount = 8
        time_moment = 0.1
        phi = np.linspace(0, 2 * np.pi * (sensor_amount - 1) / sensor_amount, sensor_amount)
        sensors = np.zeros((sensor_amount, 2), int)
        u = torch.zeros((n, n)).to(device)
        for i in range(sensor_amount):
            sensors[i, 0] = find(n, radius, radius * np.cos(-phi[i]))
            sensors[i, 1] = find(n, radius, radius * np.sin(-phi[i]))
        u = source(n, sensors, sensor_amount, 1, time_moment, 1.0)
        true_u = np.loadtxt('true_u1.txt')
        self.assertEqual(np.linalg.norm(u.cpu().numpy() - true_u), 0.0)
    
    def test_source2(self):
        n = 150
        radius = 2.0
        sensor_amount = 8
        time_moment = 0.01
        phi = np.linspace(0, 2 * np.pi * (sensor_amount - 1) / sensor_amount, sensor_amount)
        sensors = np.zeros((sensor_amount, 2), int)
        u = torch.zeros((n, n)).to(device)
        for i in range(sensor_amount):
            sensors[i, 0] = find(n, radius, radius * np.cos(-phi[i]))
            sensors[i, 1] = find(n, radius, radius * np.sin(-phi[i]))
        u = source(n, sensors, sensor_amount, 6, time_moment, 1.0)
        true_u = np.loadtxt('true_u2.txt')
        self.assertEqual(np.linalg.norm(u.cpu().numpy() - true_u), 0.0)

    def test_u0(self):
        n = 100
        radius = 2.0
        sensor_amount = 8
        time_moment = 5
        phi = np.linspace(0, 2 * np.pi * (sensor_amount - 1) / sensor_amount, sensor_amount)
        sensors = np.zeros((sensor_amount, 2), int)
        u = torch.zeros((n, n)).to(device)
        for i in range(sensor_amount):
            sensors[i, 0] = find(n, radius, radius * np.cos(-phi[i]))
            sensors[i, 1] = find(n, radius, radius * np.sin(-phi[i]))
        u=u0(u, sensors, sensor_amount, 4, time_moment)
        true_u = np.loadtxt('true_u0.txt')
        self.assertEqual(np.linalg.norm(u.cpu().numpy() - true_u), 0.0)
    
    def test_wave(self):
        n = 100
        scaling = np.ones(n * n)
        radius = 2.0
        sensor_amount = 6
        phi = np.linspace(0, 2 * np.pi * (sensor_amount - 1) / sensor_amount, sensor_amount)
        sensors = np.zeros((sensor_amount, 2), int)
        for i in range(sensor_amount):
            sensors[i, 0] = find(n, radius, radius * np.cos(-phi[i]))
            sensors[i, 1] = find(n, radius, radius * np.sin(-phi[i]))
        wave_solve(scaling, sensors, sensor_amount, radius)
        data = np.fromfile('data1.bin')
        true_data = np.fromfile('true_data1.bin')
        self.assertEqual(np.linalg.norm(data - true_data), 0.0)

    def test_read(self):
        path = '/home/farronych/20190114-185300_5B_Agar/'
        fileName = 'decode_data_01.bin'
        n_emitter = 2
        data = read_from_file(path + fileName, n_emitter)
        true_data = np.loadtxt("true_read_data.txt")
        self.assertEqual(np.linalg.norm(data - true_data), 0.0)

if __name__ == '__main__':
    main()