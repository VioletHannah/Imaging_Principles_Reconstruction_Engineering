#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/14 15:06
# @Author : 箴澄
# @Func：DAS基础实现
# @File : hw4.py
# @Software: PyCharm
"""
附件中是使用线性超声传感器阵列L7-4
    （64个传感器单元，中心频率5MHz，传感器单元两两之间中心间距0.298mm，采样频率20MHz，声速1540m/s）
通过一次全发全收采集到的数据
    （以matlab的数据文件方式存储，
    读取方法：在matlab中使用load函数直接读取，可以得到一个4096x64的矩阵数据，
            其中64对应64个传感器，4096对应每个传感器接收到的声压数据），
请用DAS算法重建图像。
提出改善成像质量的方法并对比结果。
要求
    重建图像的像素尺寸0.1mm x 0.1mm，
    重建图像宽度x深度为20mm x 40mm，
    提交包含详细注释的全部源代码和报告（源代码以压缩包形式提交，报告以pdf文件提交）。
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

Sensor_Spacing = 0.298  # mm
Sensor_Fs = 20e6 # Hz
Sound_Speed = 1540  # m/s
Pixel_Spacing = 0.1  # mm
Sensor_Perceive_Range = 4 # 假设sensor接收到的信号来自9个发射传感器单元以内的发射脉冲，即±4个单元
X = 20 // Pixel_Spacing  # 像素宽度个数
Y = 40 // Pixel_Spacing  # 像素深度个数

def distance_of_signal_flight(sensor_idxm, sensor_idxn, pixel_i, pixel_j):
    """
    计算从传感器m发射信号到达像素点(i,j)再到传感器n的总距离
    :param: 注意索引idxm, idxn, i, j都是从0开始的
    :return: 距离，单位mm
    """
    sensor_m_x = sensor_idxm * Sensor_Spacing + Sensor_Spacing / 2
    sensor_n_x = sensor_idxn * Sensor_Spacing + Sensor_Spacing / 2
    pixel_x = pixel_i * Pixel_Spacing + Pixel_Spacing / 2
    pixel_y = pixel_j * Pixel_Spacing + Pixel_Spacing / 2

    dist_m = np.sqrt((sensor_m_x - pixel_x) ** 2 + pixel_y ** 2)
    dist_n = np.sqrt((sensor_n_x - pixel_x) ** 2 + pixel_y ** 2)

    total_distance = dist_m + dist_n
    return total_distance

def time_of_signal_flight(distance):
    """
    计算距离对应的时间，实际是传感器接收到信号的采样点index
    支持标量或数组输入，返回整数索引（numpy数组或标量）
    :param distance: 距离，单位mm
    :return: 采样点index（np.ndarray 或 int）
    """
    time_s = np.asarray(distance) / 1000.0 / Sound_Speed  # s
    idx = (time_s * Sensor_Fs).astype(np.int64)
    if idx.shape == ():  # 标量
        return int(idx)
    return idx

def distance_matrix(theta_threshold=30.0):
    """
    预计算距离矩阵，减少重复计算（向量化实现）
    返回：距离(索引)矩阵，dtype=np.uint32，shape=(sizeplus, sizeplus, Y, X)
    备注：
        - 使用uint32以减少内存占用；
        - 对每个发射/接收对先计算单边距离矩阵再相加，避免像素级循环；
        - 如果像素到任一传感器连线与Y方向夹角大于30°，则该像素对该传感器对不计入（索引为0）。
    """
    size = num_sensor
    sizeplus = size + 2 * Sensor_Perceive_Range
    X_int = int(X)
    Y_int = int(Y)

    # 使用较小的整型保存索引，节省内存
    dist_mat = np.zeros((sizeplus, sizeplus, Y_int, X_int), dtype=np.uint32)

    # 预计算传感器和像素坐标（单位：mm）
    sensor_x = np.arange(sizeplus) * Sensor_Spacing + Sensor_Spacing / 2  # shape (sizeplus,)
    pixel_x = np.arange(X_int) * Pixel_Spacing + Pixel_Spacing / 2       # shape (X,)
    pixel_y = np.arange(Y_int) * Pixel_Spacing + Pixel_Spacing / 2       # shape (Y,)

    # 角度阈值（弧度）
    angle_thr = np.deg2rad(theta_threshold)

    # 为减少重复计算，对每个 m 先计算 dist_m(Y,X) 及其角度掩码，再与每个 n 的 dist_n 相加（向量化）
    for m in range(Sensor_Perceive_Range, size + Sensor_Perceive_Range):
        dx_m = pixel_x - sensor_x[m]                   # shape (X,) : px - sx
        dist_m = np.sqrt(dx_m[np.newaxis, :]**2 + pixel_y[:, np.newaxis]**2)  # shape (Y,X)

        # 计算像素到传感器 m 的角度掩码（True 表示角度 <= 30°，可接受）
        angle_m = np.arctan2(np.abs(dx_m)[np.newaxis, :], pixel_y[:, np.newaxis])  # shape (Y,X)
        mask_m = angle_m <= angle_thr  # shape (Y,X)

        for n in range(m, size + Sensor_Perceive_Range):
            dx_n = pixel_x - sensor_x[n]
            dist_n = np.sqrt(dx_n[np.newaxis, :]**2 + pixel_y[:, np.newaxis]**2)  # shape (Y,X)

            # 计算像素到传感器 n 的角度掩码
            angle_n = np.arctan2(np.abs(dx_n)[np.newaxis, :], pixel_y[:, np.newaxis])  # shape (Y,X)
            mask_n = angle_n <= angle_thr

            total_distance = dist_m + dist_n  # shape (Y,X)

            # 向量化计算时间索引并保存为 uint32
            idx = (total_distance / 1000.0 / Sound_Speed * Sensor_Fs).astype(np.uint32)

            # 若任一传感器与像素夹角超限，则不考虑该像素（置0）
            combined_mask = mask_m & mask_n
            idx[~combined_mask] = 0

            dist_mat[m, n, :, :] = idx
            dist_mat[n, m, :, :] = idx

    # np.save('distance_matrix_new.npy', dist_mat)
    print("Distance matrix saved as `distance_matrix.npy`")
    return dist_mat

def DAS_reconstruction(data, theta_threshold=30.0):
    """
    DAS算法重建图像
    :param data: 超声数据，shape=(传感器单元数, 采样点数)
    :return: 重建图像，shape=(Y, X)
    """
    # idx_matrix = np.load('distance_matrix.npy')  # 预计算距离矩阵
    idx_matrix = distance_matrix(theta_threshold)
    reconstruct_img = np.zeros((int(Y), int(X)))  # 重建图像初始化
    for i in range(int(X)):
        # dist_to_left = i * Pixel_Spacing + Pixel_Spacing / 2
        # center_sensor_idx = int((dist_to_left + Sensor_Spacing / 2) // Sensor_Spacing) # 图像点正对着的传感器idx
        # center_sensor_idx = max(center_sensor_idx, data.shape[0] - 1)
        for j in range(int(Y)):
            pixel_value = 0.0
            # if center_sensor_idx < Sensor_Spacing:
            #     for m in range(0, center_sensor_idx + Sensor_Perceive_Range + 1):  # 接收
            #         for n in range(m, center_sensor_idx + Sensor_Perceive_Range + 1): # 发射
            #             time_idx = int(idx_matrix[n, m, j, i])
            #             if time_idx < data.shape[1]:
            #                 pixel_value += data[m, time_idx] + data[n, time_idx]
            # elif center_sensor_idx > data.shape[0] - Sensor_Perceive_Range - 1:
            #     for m in range(center_sensor_idx - Sensor_Perceive_Range, data.shape[0]):  # 接收
            #         for n in range(m, data.shape[0]): # 发射
            #             time_idx = int(idx_matrix[n, m, j, i])
            #             if time_idx < data.shape[1]:
            #                 pixel_value += data[m, time_idx] + data[n, time_idx]
            # else:
            #     for m in range(center_sensor_idx - Sensor_Perceive_Range, center_sensor_idx + Sensor_Perceive_Range + 1):  # 接收
            #         for n in range(m, center_sensor_idx + Sensor_Perceive_Range + 1): # 发射
            #             time_idx = int(idx_matrix[n, m, j, i])
            #             if time_idx < data.shape[1]:
            #                 pixel_value += data[m, time_idx] + data[n, time_idx]
            for tx in range(data.shape[0]):
                for rx in range(tx, data.shape[0]):
                    time_idx = int(idx_matrix[tx, rx, j, i])
                    if time_idx < data.shape[1]:
                        pixel_value += data[tx, time_idx] + data[rx, time_idx]
            reconstruct_img[j, i] = pixel_value
    return reconstruct_img

def main(output_path='hw4'):
    os.makedirs(output_path, exist_ok=True)
    # 读取数据文件，已经提前预存为npy格式
    from hw4_enhanced import preprocess_signal
    data = np.load('b_data.npy')
    ultrasound_data = preprocess_signal(data)
    print(f'Ultrasound data shape: {ultrasound_data.shape}') # (64, 4096)
    global num_sensor # 声明全局变量
    num_sensor, num_samples = ultrasound_data.shape
    dlc = np.zeros([Sensor_Perceive_Range, num_samples])
    ultrasound_data = np.vstack((dlc, ultrasound_data, dlc))  # 在上下各添加4行0，变为72x4096
    print(f'Padded ultrasound data shape: {ultrasound_data.shape}')

    # 执行DAS重建
    print(f'Starting DAS reconstruction...')
    reconstructed_img = [DAS_reconstruction(ultrasound_data, theta) for theta in np.linspace(15, 90, num=6)]

    # 显示重建图像
    for img in reconstructed_img:
        plt.imshow(img, cmap='gray', extent=(0, img.shape[1] * Pixel_Spacing, img.shape[0] * Pixel_Spacing, 0), origin='upper', aspect='auto')
        plt.xlabel('Width (mm)')
        plt.ylabel('Depth (mm)')
        plt.title('DAS Reconstructed Image')
        plt.colorbar(label='Intensity')
        plt.show()

    # 保存重建图像
    for i, theta in enumerate(np.linspace(15, 90, num=6)):
        path = os.path.join(output_path, f'DAS_theta_{theta}.png')
        cv2.imwrite(path, (reconstructed_img[i] / np.max(reconstructed_img[i]) * 255).astype(np.uint8))
    print('Reconstructed image saved as DAS_reconstructed_image.png')

if __name__ == '__main__':
    main()