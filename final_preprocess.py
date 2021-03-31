import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

possible_range = 100
matrix_size = [20, 10]
time_slot = 200
time_interval = 0.01
threshold = 1
window_size = 3
domain = int((window_size - 1) / 2)
back_bias = 0.1 * time_slot

zip_path = "D:\\SUSTC\\2020-2021·S5\\创新实践I\\2021data\\"
output_path = "C:\\Users\\THINKPAD\\PycharmProjects\\untitled2\\"
tool_path = "C:\\Users\\THINKPAD\\PycharmProjects\\untitled2\\temp\\"
all_kind = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'S', 'CIRCLE']

def test(table, index):
    count = 0
    for i in range(possible_range):
        if index + i >= len(table) - 1:
            break
        x = pow(table["Linear Acceleration x (m/s^2)"][index + i], 2)
        y = pow(table["Linear Acceleration y (m/s^2)"][index + i], 2)
        z = pow(table["Linear Acceleration z (m/s^2)"][index + i], 2)

        absolute = pow(x + y + z, 0.5)
        if absolute > threshold:
            count += 1
    return count > possible_range / 2

def is_time_to_break(gyr, index, count, time):
    for i in range(0, len(gyr)):
        if gyr["Time (s)"][i] >= time:
            return i + time_slot > len(gyr)
    return True

def gyr_start_index(gyr, la, index):
    time = la['Time (s)'][index]
    for i in range(len(gyr)):
        if gyr['Time (s)'][i] >= time:
            return i
    return 0

if __name__ == '__main__':
    counter = 0
    # print(len(os.listdir(zip_path)))
    for zip_file in os.listdir(zip_path):
        # print(zip_file)
        readable_file = zipfile.ZipFile(zip_path + zip_file, "r")
        kind = zip_file.split('-')[0]
        # print(kind, all_kind.index(kind))
        gyroscopes = None
        accelerations = None
        for file in readable_file.namelist():
            if file.__contains__("Linear Acceleration"):
                f = readable_file.open(file)
                accelerations = pd.read_csv(f)
                print(accelerations)
            elif file.__contains__("Gyroscope"):
                f = readable_file.open(file)
                gyroscopes = pd.read_csv(f)
                print(gyroscopes)
        # smooth data, window == 5
        for j in range(domain, len(accelerations) - domain):
            accelerations['Linear Acceleration x (m/s^2)'][j] = \
                accelerations['Linear Acceleration x (m/s^2)'][j - domain: j + domain].sum() / window_size
            accelerations['Linear Acceleration y (m/s^2)'][j] = \
                accelerations['Linear Acceleration y (m/s^2)'][j - domain: j + domain].sum() / window_size
            accelerations['Linear Acceleration z (m/s^2)'][j] = \
                accelerations['Linear Acceleration z (m/s^2)'][j - domain: j + domain].sum() / window_size

        for j in range(domain, len(gyroscopes) - domain):
            gyroscopes['Gyroscope x (rad/s)'][j] = \
                gyroscopes['Gyroscope x (rad/s)'][j - domain: j + domain].sum() / window_size
            gyroscopes['Gyroscope y (rad/s)'][j] = \
                gyroscopes['Gyroscope y (rad/s)'][j - domain: j + domain].sum() / window_size
            gyroscopes['Gyroscope z (rad/s)'][j] = \
                gyroscopes['Gyroscope z (rad/s)'][j - domain: j + domain].sum() / window_size

        temp = []
        chosen_la  = []
        chosen_gyr = []
        tempcount = 0
        gyrcount = 0
        for i in range(len(accelerations)):
            # print(i)
            if tempcount != 0:
                tempcount -= 1
                continue
            else:
                tempcount = 0
            x = pow(accelerations["Linear Acceleration x (m/s^2)"][i], 2)
            y = pow(accelerations["Linear Acceleration y (m/s^2)"][i], 2)
            z = pow(accelerations["Linear Acceleration z (m/s^2)"][i], 2)
            absolute = pow(x + y + z, 0.5)
            # print(absolute)
            if absolute > threshold and test(accelerations, i):
                if is_time_to_break(gyroscopes, i, gyrcount, accelerations["Time (s)"][i]):
                    # print('break')
                    break
                else:
                    gyrcount = i
                # print(accelerations["Time (s)"][i])
                temp.append(accelerations["Time (s)"][i] - back_bias * 0.01)
                if i >= back_bias:
                    chosen_la.append(i - back_bias)
                    chosen_gyr.append(gyr_start_index(gyroscopes, accelerations, i - back_bias))
                tempcount = time_slot
        # print(temp)
        # print(len(temp))
        for i in range(len(chosen_la) - 1):
            # linear matrix
            #shape: 1*201,last one is label
            matrixlax = np.zeros((1, matrix_size[0] * matrix_size[1] + 1))
            matrixlay = np.zeros((1, matrix_size[0] * matrix_size[1] + 1))
            matrixlaz = np.zeros((1, matrix_size[0] * matrix_size[1] + 1))
            # gyr matrix
            matrixgyx = np.zeros((1, matrix_size[0] * matrix_size[1] + 1))
            matrixgyy = np.zeros((1, matrix_size[0] * matrix_size[1] + 1))
            matrixgyz = np.zeros((1, matrix_size[0] * matrix_size[1] + 1))
            la_start = chosen_la[i]
            gyr_start = chosen_gyr[i]
            for j in range(time_slot):
                matrixlax[0][j] = accelerations["Linear Acceleration x (m/s^2)"][la_start + j]
                matrixlay[0][j] = accelerations["Linear Acceleration y (m/s^2)"][la_start + j]
                matrixlaz[0][j] = accelerations["Linear Acceleration z (m/s^2)"][la_start + j]
                matrixgyx[0][j] = gyroscopes['Gyroscope x (rad/s)'][gyr_start + j]
                matrixgyy[0][j] = gyroscopes['Gyroscope y (rad/s)'][gyr_start + j]
                matrixgyz[0][j] = gyroscopes['Gyroscope z (rad/s)'][gyr_start + j]
            current_label = all_kind.index(kind)
            matrixlax[0][time_slot] = current_label
            matrixlay[0][time_slot] = current_label
            matrixlaz[0][time_slot] = current_label
            matrixgyx[0][time_slot] = current_label
            matrixgyy[0][time_slot] = current_label
            matrixgyz[0][time_slot] = current_label
            filenameX = tool_path + 'laX.csv'
            filenameY = tool_path + 'laY.csv'
            filenameZ = tool_path + 'laZ.csv'
            np.savetxt(filenameX, np.asarray(matrixlax), delimiter=",")
            np.savetxt(filenameY, np.asarray(matrixlay), delimiter=",")
            np.savetxt(filenameZ, np.asarray(matrixlaz), delimiter=",")
            filenameX = tool_path + 'gyX.csv'
            filenameY = tool_path + 'gyY.csv'
            filenameZ = tool_path + 'gyZ.csv'
            np.savetxt(filenameX, np.asarray(matrixgyx), delimiter=",")
            np.savetxt(filenameY, np.asarray(matrixgyy), delimiter=",")
            np.savetxt(filenameZ, np.asarray(matrixgyz), delimiter=",")
            with open(output_path + 'gyX_train.csv', 'ab') as f:
                f.write(open(tool_path + 'gyX.csv', 'rb').read())
            with open(output_path + 'gyY_train.csv', 'ab') as f:
                f.write(open(tool_path + 'gyY.csv', 'rb').read())
            with open(output_path + 'gyZ_train.csv', 'ab') as f:
                f.write(open(tool_path + 'gyZ.csv', 'rb').read())
            with open(output_path + 'laX_train.csv', 'ab') as f:
                f.write(open(tool_path + 'laX.csv', 'rb').read())
            with open(output_path + 'laY_train.csv', 'ab') as f:
                f.write(open(tool_path + 'laY.csv', 'rb').read())
            with open(output_path + 'laZ_train.csv', 'ab') as f:
                f.write(open(tool_path + 'laZ.csv', 'rb').read())
        counter += 1
        print("finished: ", kind, ". Left: ", counter, " / ", len(os.listdir(zip_path)))