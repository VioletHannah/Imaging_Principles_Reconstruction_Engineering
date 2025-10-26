#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超声成像DAS (Delay-and-Sum) 算法重建 - 优化版本
L7-4线性阵列: 64传感器, 5MHz中心频率, 0.298mm间距, 20MHz采样, 1540m/s声速
重建参数: 0.1mm分辨率, 20mm x 40mm成像区域
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import cv2
import time

# 系统参数
SENSOR_SPACING = 0.298e-3  # 传感器间距 (m)
SAMPLING_FREQ = 20e6  # 采样频率 (Hz)
SOUND_SPEED = 1540  # 声速 (m/s)
CENTER_FREQ = 5e6  # 中心频率 (Hz)
NUM_SENSORS = 64  # 传感器数量

# 成像参数
PIXEL_SIZE = 0.1e-3  # 像素尺寸 (m)
IMAGE_WIDTH = 20e-3  # 成像宽度 (m)
IMAGE_DEPTH = 40e-3  # 成像深度 (m)

# 计算像素数量
NUM_X = int(IMAGE_WIDTH / PIXEL_SIZE)  # 200像素
NUM_Y = int(IMAGE_DEPTH / PIXEL_SIZE)  # 400像素


def load_ultrasound_data(filename='b_data.npy'):
    """加载超声数据"""
    try:
        data = np.load(filename)
        print(f'加载数据: {data.shape} (传感器 × 采样点)')
        return data
    except FileNotFoundError:
        print(f'文件 {filename} 不存在')

def preprocess_signal(data, apply_filter=True, apply_hilbert=True):
    """
    信号预处理
    :param data: 原始RF数据 (num_sensors, num_samples)
    :param apply_filter: 是否应用带通滤波
    :param apply_hilbert: 是否应用希尔伯特变换提取包络
    :return: 预处理后的数据
    """
    processed_data = data.copy()

    if apply_filter:
        # 带通滤波器: 中心频率±50%
        nyquist = SAMPLING_FREQ / 2
        low_freq = CENTER_FREQ / np.sqrt(2) / nyquist
        high_freq = min(CENTER_FREQ * np.sqrt(2) / nyquist, 0.99)
        band = np.array([low_freq, high_freq])
        sos = signal.butter(4, band, 'band', output='sos')
        processed_data = signal.sosfiltfilt(sos, processed_data, axis=1)
        print('已应用带通滤波器')

    if apply_hilbert:
        # 希尔伯特变换提取包络
        analytic_signal = signal.hilbert(processed_data, axis=1)
        processed_data = np.abs(analytic_signal)
        print('应用希尔伯特变换提取包络')

    return processed_data

def das_reconstruction_vectorized(data, apodization=True, dynamic_range_db=60, apt=4):
    """
    向量化DAS重建算法（全发全收模式）
    :param data: 超声RF数据 (num_sensors, num_samples)
    :param apodization: 是否应用窗函数
    :param dynamic_range_db: 动态范围(dB)
    :return: 重建图像 (num_y, num_x)
    """
    num_sensors, num_samples = data.shape
    image = np.zeros((NUM_Y, NUM_X))

    # 创建像素坐标网格
    pixel_x = np.arange(NUM_X) * PIXEL_SIZE + PIXEL_SIZE / 2  # (NUM_X,)
    pixel_y = np.arange(NUM_Y) * PIXEL_SIZE + PIXEL_SIZE / 2  # (NUM_Y,)

    # 传感器坐标
    sensor_x = np.arange(num_sensors) * SENSOR_SPACING + SENSOR_SPACING / 2  # (num_sensors,)

    print(f'开始DAS重建: {NUM_X}×{NUM_Y}像素...')
    start_time = time.time()

    # 逐深度层重建（避免内存溢出）
    for j in range(NUM_Y):
        y = pixel_y[j]

        # 计算所有传感器到所有像素的距离
        # sensor_x: (64,) -> (64, 1)
        # pixel_x: (200,) -> (1, 200)
        dx = sensor_x[:, np.newaxis] - pixel_x[np.newaxis, :]  # (64, 200) 第i个sensor到第j个像素的水平距离
        distances = np.sqrt(dx ** 2 + y ** 2)  # (64, 200) 第i个sensor到第j个像素的距离

        # 全发全收: 每个传感器既发射又接收
        # 对于像素(i,j)，来自所有传感器对的贡献
        for tx_idx in range(num_sensors):
            # 发射传感器到像素的距离
            tx_distances = distances[tx_idx, :]  # (200,)

            # 所有接收传感器到像素的距离
            rx_distances = distances  # (64, 200)

            # 总飞行时间对应的采样点索引
            total_distances = tx_distances[np.newaxis, :] + rx_distances  # (64, 200)
            time_delays = total_distances / SOUND_SPEED  # (64, 200)
            sample_indices = (time_delays * SAMPLING_FREQ).astype(int)  # (64, 200)

            # 边界检查
            valid_mask = (sample_indices >= 0) & (sample_indices < num_samples)

            # 提取对应采样点的值
            for rx_idx in range(num_sensors):
                valid = valid_mask[rx_idx, :]
                idx = sample_indices[rx_idx, valid]

                contribution = np.zeros(NUM_X)
                contribution[valid] = data[rx_idx, idx]

                # 应用窗函数（距离加权）
                if apodization:
                    # 使用汉宁窗进行孔径加权
                    lateral_distance = np.abs(pixel_x - sensor_x[tx_idx])
                    aperture = IMAGE_WIDTH / apt  # 有效孔径
                    weight = np.where(lateral_distance < aperture,
                                      np.cos(np.pi * lateral_distance / (2 * aperture)) ** 2,
                                      0)
                    contribution *= weight

                image[j, :] += contribution

    elapsed = time.time() - start_time
    print(f'重建完成，耗时: {elapsed:.2f}秒')

    # 对数压缩和动态范围调整
    image = np.abs(image)
    image_db = 20 * np.log10(image + 1e-10)  # 避免log(0)
    image_db -= np.max(image_db)  # 归一化到0dB
    # image_db = np.clip(image_db, -dynamic_range_db, 0)  # 限制动态范围

    return image_db


def das_reconstruction_optimized(data, apodization=True, dynamic_range_db=60):
    """
    优化的DAS重建（减少循环，提高效率）
    采用合成孔径聚焦技术(SAFT)思想
    """
    num_sensors, num_samples = data.shape
    image = np.zeros((NUM_Y, NUM_X))

    # 像素坐标
    pixel_x = np.arange(NUM_X) * PIXEL_SIZE + PIXEL_SIZE / 2
    pixel_y = np.arange(NUM_Y) * PIXEL_SIZE + PIXEL_SIZE / 2

    # 传感器坐标
    sensor_x = np.arange(num_sensors) * SENSOR_SPACING + SENSOR_SPACING / 2

    print(f'开始优化DAS重建: {NUM_X}×{NUM_Y}像素...')
    start_time = time.time()

    # 预计算传感器到所有像素的距离（一次性计算）
    X_grid, Y_grid = np.meshgrid(pixel_x, pixel_y)  # (NUM_Y, NUM_X)

    for sensor_idx in range(num_sensors):
        if sensor_idx % 16 == 0:
            print(f'  处理传感器: {sensor_idx}/{num_sensors}')

        sx = sensor_x[sensor_idx]

        # 该传感器到所有像素的距离
        dist_to_pixels = np.sqrt((X_grid - sx) ** 2 + Y_grid ** 2)  # (NUM_Y, NUM_X)

        # 全发全收：该传感器作为发射器和接收器
        # 双程距离 = 2 * 单程距离（发射+反射+接收）
        round_trip_time = 2 * dist_to_pixels / SOUND_SPEED
        sample_indices = (round_trip_time * SAMPLING_FREQ).astype(int)

        # 边界检查并累加
        valid_mask = (sample_indices >= 0) & (sample_indices < num_samples)
        valid_indices = sample_indices[valid_mask]

        contribution = np.zeros_like(image)
        contribution[valid_mask] = data[sensor_idx, valid_indices]

        # 应用孔径加权
        if apodization:
            lateral_dist = np.abs(X_grid - sx) # (NUM_Y, NUM_X)
            aperture = IMAGE_WIDTH / 4
            weight = np.where(lateral_dist < aperture,
                              np.cos(np.pi * lateral_dist / (2 * aperture)) ** 2,
                              0)
            contribution *= weight

        image += contribution

    elapsed = time.time() - start_time
    print(f'优化重建完成，耗时: {elapsed:.2f}秒')

    # 后处理
    image = np.abs(image)
    image_db = 20 * np.log10(image + 1e-10)
    image_db -= np.max(image_db)
    image_db = np.clip(image_db, -dynamic_range_db, 0)

    return image_db


def enhance_image(image, method='clahe'):
    """
    图像增强
    :param image: 输入图像(dB)
    :param method: 增强方法 ('clahe', 'gamma', 'none')
    :return: 增强后的图像
    """
    # 归一化到0-255
    img_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    if method == 'clahe':
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)
        print('应用CLAHE增强')
    elif method == 'gamma':
        # Gamma校正
        gamma = 0.5
        enhanced = np.power(img_norm / 255.0, gamma) * 255
        enhanced = enhanced.astype(np.uint8)
        print(f'应用Gamma校正 (γ={gamma})')
    else:
        enhanced = img_norm

    return enhanced


def visualize_results(original_img, enhanced_img, save_fig=True):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原始DAS重建
    im1 = axes[0].imshow(original_img, cmap='gray', aspect='auto',
                         extent=[0, IMAGE_WIDTH * 1000, IMAGE_DEPTH * 1000, 0])
    axes[0].set_title('DAS重建图像', fontsize=14, fontproperties='SimHei')
    axes[0].set_xlabel('宽度 (mm)', fontproperties='SimHei')
    axes[0].set_ylabel('深度 (mm)', fontproperties='SimHei')
    plt.colorbar(im1, ax=axes[0], label='强度 (dB)')

    # 增强后图像
    im2 = axes[1].imshow(enhanced_img, cmap='gray', aspect='auto',
                         extent=[0, IMAGE_WIDTH * 1000, IMAGE_DEPTH * 1000, 0])
    axes[1].set_title('增强后图像 (CLAHE)', fontsize=14, fontproperties='SimHei')
    axes[1].set_xlabel('宽度 (mm)', fontproperties='SimHei')
    axes[1].set_ylabel('深度 (mm)', fontproperties='SimHei')
    plt.colorbar(im2, ax=axes[1], label='强度')

    plt.tight_layout()

    if save_fig:
        plt.savefig('ultrasound_comparison.png', dpi=300, bbox_inches='tight')
        print('✓ 对比图保存为: ultrasound_comparison.png')

    plt.show()


def main(output_dir='results'):
    """主函数"""
    print('=' * 60)
    print('超声成像DAS重建程序')
    print('=' * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据
    raw_data = load_ultrasound_data('b_data.npy')

    # 2. 信号预处理
    processed_data = preprocess_signal(raw_data,
                                       apply_filter=True,
                                       apply_hilbert=True)

    # 3. DAS重建
    reconstructed_ori = das_reconstruction_vectorized(raw_data, apodization=True)

    reconstructed_img = [das_reconstruction_vectorized(
        processed_data,
        apodization=True,
        dynamic_range_db=60,
        apt=i
    ) for i in range(2, 6)]

    reconstructed_single

    # 4. 对比不同图像增强
    enhanced_img_clahe = enhance_image(reconstructed_img[3], method='clahe')
    enhanced_img_gamma = enhance_image(reconstructed_img[3], method='gamma')

    # 5. 保存各个结果
    for i, reconstructed in enumerate(reconstructed_img):
        out_path = os.path.join(output_dir, f'das_processed_apt={i+2}.png')
        norm = ((reconstructed - reconstructed.min()) /
                (reconstructed.max() - reconstructed.min()) * 255).astype(np.uint8)
        cv2.imwrite(out_path, norm)
    norm = ((reconstructed_ori - reconstructed_ori.min()) /
            (reconstructed_ori.max() - reconstructed_ori.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'das_original.png'), norm)
    cv2.imwrite(os.path.join(output_dir, 'das_enhanced_clahe.png'), enhanced_img_clahe)
    cv2.imwrite(os.path.join(output_dir, 'das_enhanced_gamma.png'), enhanced_img_gamma)
    print('图像已保存')

    # 6. 可视化对比
    # visualize_results(reconstructed_img, enhanced_img)

    print('=' * 60)
    print('处理完成！')
    print('=' * 60)


if __name__ == '__main__':
    main()