import numpy as np

elements_num = 2048
time_points_num = 3750

#часть ани леоновой
fileName1 = 'decode_data_01.bin'
fileName2 = 'decode_data_02.bin'
fileName3 = 'decode_data_03.bin'
fileName4 = 'decode_data_04.bin'
outputFilename = 'each_8.txt'

def read_from_file(fileName, n_emitter):
    offset = int((elements_num / 4) * time_points_num * (n_emitter - 1) * 2)
    with open(fileName, mode='rb') as file:
        file.seek(offset, 0)
        file_content = file.read(time_points_num * elements_num // 4 * 2)
        test_data = np.frombuffer(file_content, dtype=np.int16)
        data = np.reshape(test_data, (time_points_num, elements_num // 4))
        data = np.transpose(data)
    return data
#--------------------
sensor_count=2048
i=4
ticks=3750
dtype = np.dtype('B')
f=open('data_breast.bin', 'rb')
numpy_data = np.atleast_1d(np.fromfile(f,dtype))
left_edge=(i-1)*ticks*sensor_count**2
right_edge=i*ticks*sensor_count**2
id_emit_diagram=numpy_data[left_edge:right_edge].astype(dtype)
id_emit_diagram=np.reshape(ticks, sensor_count)
a=1