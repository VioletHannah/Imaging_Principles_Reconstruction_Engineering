#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/26 12:55
# @Author : 箴澄
# @Func：改进基于几何角度的动态孔径聚焦
# @File : hw4_apodization_enhanced.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from hw4_enhanced import load_ultrasound_data, preprocess_signal, enhance_image
import cv2
import time

# ==================== 系统参数 ====================
SENSOR_SPACING = 0.298e-3  # 传感器间距 (m)
SAMPLING_FREQ = 20e6  # 采样频率 (Hz)
SOUND_SPEED = 1540  # 声速 (m/s)
CENTER_FREQ = 5e6  # 中心频率 (Hz)
NUM_SENSORS = 64  # 传感器数量

# ==================== 成像参数 ====================
PIXEL_SIZE = 0.1e-3  # 像素尺寸 (m)
IMAGE_WIDTH = 20e-3  # 成像宽度 (m)
IMAGE_DEPTH = 40e-3  # 成像深度 (m)

NUM_X = int(IMAGE_WIDTH / PIXEL_SIZE)  # 200像素
NUM_Y = int(IMAGE_DEPTH / PIXEL_SIZE)  # 400像素

def angular_apodization(lateral_dist, depth, angle_threshold_deg=60,
                        apod_type='hanning', smooth_transition=True):
    """
    基于角度的动态孔径加权

    :param lateral_dist: 横向距离（像素到传感器的X方向距离）(m)
    :param depth: 深度（像素的Y坐标）(m)
    :param angle_threshold_deg: 角度阈值（度），默认60度（tan(60°)=√3）
    :param apod_type: 加权类型 ('hanning', 'hamming', 'gaussian', 'linear')
    :param smooth_transition: 是否平滑过渡
    :return: 权重数组
    """
    # 避免除零
    depth = np.maximum(depth, 1e-6)

    # 计算角度比值: tan(θ) = lateral_dist / depth
    tan_angle = lateral_dist / depth

    # 角度阈值对应的tan值
    tan_threshold = np.tan(np.deg2rad(angle_threshold_deg))

    # 初始化权重
    weight = np.zeros_like(lateral_dist)

    if smooth_transition:
        # 平滑过渡：在阈值附近使用余弦窗
        transition_ratio = 0.8  # 80%处开始衰减
        tan_transition = tan_threshold * transition_ratio

        # 完全接收区域 (tan_angle < tan_transition)
        full_region = tan_angle < tan_transition

        # 过渡区域 (tan_transition <= tan_angle < tan_threshold)
        transition_region = (tan_angle >= tan_transition) & (tan_angle < tan_threshold)

        # 拒绝区域 (tan_angle >= tan_threshold)
        # 权重已经是0，不需要处理

        # 完全接收区域根据apod_type设置权重
        if apod_type == 'hanning':
            # Hanning窗：w(x) = 0.5 * (1 + cos(π*x/x_max))
            normalized_pos = tan_angle[full_region] / tan_transition
            weight[full_region] = 0.5 * (1 + np.cos(np.pi * normalized_pos))

        elif apod_type == 'hamming':
            # Hamming窗
            normalized_pos = tan_angle[full_region] / tan_transition
            weight[full_region] = 0.54 + 0.46 * np.cos(np.pi * normalized_pos)

        elif apod_type == 'gaussian':
            # 高斯窗
            sigma = tan_transition / 3  # 3-sigma准则
            weight[full_region] = np.exp(-0.5 * (tan_angle[full_region] / sigma) ** 2)

        elif apod_type == 'linear':
            # 线性衰减
            weight[full_region] = 1 - tan_angle[full_region] / tan_transition

        else:  # 'uniform'
            weight[full_region] = 1.0

        # 过渡区域：余弦平滑下降到0
        if np.any(transition_region):
            transition_normalized = (tan_angle[transition_region] - tan_transition) / (tan_threshold - tan_transition)
            weight[transition_region] = weight[full_region].max() * 0.5 * (1 + np.cos(np.pi * transition_normalized))

    else:
        # 硬截断：角度内权重为1，角度外权重为0
        valid_region = tan_angle < tan_threshold
        weight[valid_region] = 1.0

    return weight


def das_reconstruction_single_monostatic(data, angle_threshold=60, apod_type='hanning',
                                         smooth_transition=True, dynamic_range_db=60):
    """
    单站模式DAS重建（每个传感器自己发射自己接收）
    采用改进的角度孔径加权

    :param data: 脉冲回波数据 (num_sensors, num_samples)
    :param angle_threshold: 角度阈值（度）
    :param apod_type: 加权窗类型
    :param smooth_transition: 是否平滑过渡
    :param dynamic_range_db: 动态范围(dB)
    :return: 重建图像 (num_y, num_x)
    """
    num_sensors, num_samples = data.shape
    image = np.zeros((NUM_Y, NUM_X))

    pixel_x = np.arange(NUM_X) * PIXEL_SIZE + PIXEL_SIZE / 2
    pixel_y = np.arange(NUM_Y) * PIXEL_SIZE + PIXEL_SIZE / 2
    sensor_x = np.arange(num_sensors) * SENSOR_SPACING + SENSOR_SPACING / 2

    print(f'开始单站DAS重建（角度孔径: {angle_threshold}°）...')
    start_time = time.time()

    X_grid, Y_grid = np.meshgrid(pixel_x, pixel_y)

    for sensor_idx in range(num_sensors):
        if sensor_idx % 16 == 0:
            print(f'  处理传感器: {sensor_idx}/{num_sensors}')

        sx = sensor_x[sensor_idx]

        # 计算横向距离和深度
        lateral_dist = np.abs(X_grid - sx)  # (NUM_Y, NUM_X)
        depth = Y_grid  # (NUM_Y, NUM_X)

        # 基于角度的动态孔径加权
        aperture_weight = angular_apodization(
            lateral_dist,
            depth,
            angle_threshold_deg=angle_threshold,
            apod_type=apod_type,
            smooth_transition=smooth_transition
        )

        # 计算双程距离和时延
        dist_to_pixels = np.sqrt(lateral_dist ** 2 + depth ** 2)
        round_trip_time = 2 * dist_to_pixels / SOUND_SPEED
        sample_indices = (round_trip_time * SAMPLING_FREQ).astype(int)

        # 边界检查
        valid_mask = (sample_indices >= 0) & (sample_indices < num_samples)
        valid_indices = sample_indices[valid_mask]

        # 累加贡献
        contribution = np.zeros_like(image)
        contribution[valid_mask] = data[sensor_idx, valid_indices]

        # 应用角度孔径权重
        contribution *= aperture_weight

        image += contribution

    elapsed = time.time() - start_time
    print(f'✓ 单站重建完成，耗时: {elapsed:.2f}秒')

    # 后处理
    image = np.abs(image)
    image_db = 20 * np.log10(image + 1e-10)
    image_db -= np.max(image_db)
    image_db = np.clip(image_db, -dynamic_range_db, 0)

    return image_db


def das_reconstruction_full_bistatic(data, angle_threshold=60, apod_type='hanning',
                                     smooth_transition=True, dynamic_range_db=60):
    """
    全发全收双站模式DAS重建（假设数据包含所有传感器对）
    采用改进的角度孔径加权

    注意：此函数假设data格式为 (num_sensors, num_samples)，
    其中每个传感器的数据包含它作为接收器时接收到的所有信号

    :param data: 超声数据 (num_sensors, num_samples)
    :param angle_threshold: 角度阈值（度）
    :param apod_type: 加权窗类型
    :param smooth_transition: 是否平滑过渡
    :param dynamic_range_db: 动态范围(dB)
    :return: 重建图像 (num_y, num_x)
    """
    num_sensors, num_samples = data.shape
    image = np.zeros((NUM_Y, NUM_X))

    pixel_x = np.arange(NUM_X) * PIXEL_SIZE + PIXEL_SIZE / 2
    pixel_y = np.arange(NUM_Y) * PIXEL_SIZE + PIXEL_SIZE / 2
    sensor_x = np.arange(num_sensors) * SENSOR_SPACING + SENSOR_SPACING / 2

    print(f'开始全发全收DAS重建（角度孔径: {angle_threshold}°）...')
    start_time = time.time()

    X_grid, Y_grid = np.meshgrid(pixel_x, pixel_y)

    pair_count = 0
    total_pairs = num_sensors * num_sensors

    # 遍历所有发射-接收传感器对
    for tx_idx in range(num_sensors):
        for rx_idx in range(num_sensors):
            pair_count += 1
            if pair_count % 500 == 0:
                print(f'  处理传感器对: {pair_count}/{total_pairs} ({pair_count / total_pairs * 100:.1f}%)')

            tx_x = sensor_x[tx_idx]
            rx_x = sensor_x[rx_idx]

            # 计算距离
            lateral_dist_tx = np.abs(X_grid - tx_x)
            lateral_dist_rx = np.abs(X_grid - rx_x)
            depth = Y_grid

            dist_tx = np.sqrt(lateral_dist_tx ** 2 + depth ** 2)
            dist_rx = np.sqrt(lateral_dist_rx ** 2 + depth ** 2)
            total_dist = dist_tx + dist_rx

            # 发射和接收的角度孔径加权
            weight_tx = angular_apodization(
                lateral_dist_tx, depth,
                angle_threshold_deg=angle_threshold,
                apod_type=apod_type,
                smooth_transition=smooth_transition
            )

            weight_rx = angular_apodization(
                lateral_dist_rx, depth,
                angle_threshold_deg=angle_threshold,
                apod_type=apod_type,
                smooth_transition=smooth_transition
            )

            # 综合权重（发射和接收权重的乘积）
            combined_weight = weight_tx * weight_rx

            # 时延计算
            time_delay = total_dist / SOUND_SPEED
            sample_indices = (time_delay * SAMPLING_FREQ).astype(int)

            # 边界检查
            valid_mask = (sample_indices >= 0) & (sample_indices < num_samples)
            valid_indices = sample_indices[valid_mask]

            # 提取信号并累加
            contribution = np.zeros_like(image)
            contribution[valid_mask] = data[rx_idx, valid_indices]

            # 应用综合权重
            contribution *= combined_weight

            image += contribution

    elapsed = time.time() - start_time
    print(f'✓ 全发全收重建完成，耗时: {elapsed:.2f}秒')

    # 后处理
    image = np.abs(image)
    image_db = 20 * np.log10(image + 1e-10)
    image_db -= np.max(image_db)
    image_db = np.clip(image_db, -dynamic_range_db, 0)

    return image_db


def visualize_comparison(images_dict, titles_dict, save_name='comparison.png'):
    """
    可视化多个重建结果对比

    :param images_dict: {key: image_array} 字典
    :param titles_dict: {key: title_string} 字典
    :param save_name: 保存文件名
    """
    n_images = len(images_dict)
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))

    if n_images == 1:
        axes = [axes]

    for idx, (key, image) in enumerate(images_dict.items()):
        im = axes[idx].imshow(image, cmap='gray', aspect='auto',
                              extent=[0, IMAGE_WIDTH * 1000, IMAGE_DEPTH * 1000, 0])
        axes[idx].set_title(titles_dict[key], fontsize=12, fontproperties='SimHei')
        axes[idx].set_xlabel('宽度 (mm)', fontproperties='SimHei')
        axes[idx].set_ylabel('深度 (mm)', fontproperties='SimHei')
        plt.colorbar(im, ax=axes[idx], label='强度 (dB)')

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f'✓ 对比图保存为: {save_name}')
    plt.show()


def visualize_aperture_weights():
    """可视化不同角度阈值和加权类型的孔径权重分布"""
    depths = np.linspace(0.001, 0.040, 400)  # 1-40mm深度
    lateral_dists = np.linspace(0, 0.010, 100)  # 0-10mm横向距离

    D, L = np.meshgrid(depths, lateral_dists)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    configs = [
        (60, 'hanning', True),
        (60, 'linear', True),
        (60, 'gaussian', True),
        (45, 'hanning', True),
        (60, 'hanning', False),
        (75, 'hanning', True),
    ]

    titles = [
        '60° Hanning (平滑)',
        '60° Linear (平滑)',
        '60° Gaussian (平滑)',
        '45° Hanning (平滑)',
        '60° Hanning (硬截断)',
        '75° Hanning (平滑)',
    ]

    for idx, (ax, (angle, apod, smooth), title) in enumerate(zip(axes.flat, configs, titles)):
        weights = angular_apodization(L, D, angle, apod, smooth)

        im = ax.contourf(D * 1000, L * 1000, weights, levels=20, cmap='viridis')
        ax.set_xlabel('深度 (mm)', fontproperties='SimHei')
        ax.set_ylabel('横向距离 (mm)', fontproperties='SimHei')
        ax.set_title(title, fontproperties='SimHei')
        plt.colorbar(im, ax=ax, label='权重')

        # 绘制角度线
        tan_val = np.tan(np.deg2rad(angle))
        angle_line_d = depths * 1000
        angle_line_l = tan_val * depths * 1000
        mask = angle_line_l <= 10
        ax.plot(angle_line_d[mask], angle_line_l[mask], 'r--', linewidth=2, label=f'{angle}°线')
        ax.legend()

    plt.tight_layout()
    plt.savefig('aperture_weights_visualization.png', dpi=300, bbox_inches='tight')
    print('✓ 孔径权重可视化保存为: aperture_weights_visualization.png')
    plt.show()


def main():
    """主函数 - 对比不同重建方法"""
    print('=' * 70)
    print('超声成像DAS重建 - 改进孔径聚焦对比实验')
    print('=' * 70)

    # 1. 可视化孔径权重
    print('\n[步骤1] 可视化孔径权重分布...')
    visualize_aperture_weights()

    # 2. 加载数据
    print('\n[步骤2] 加载超声数据...')
    raw_data = load_ultrasound_data('b_data.npy')

    # 3. 信号预处理
    print('\n[步骤3] 信号预处理...')
    processed_data = preprocess_signal(raw_data, apply_filter=True, apply_hilbert=True)

    # 4. 不同方法重建对比
    print('\n[步骤4] 执行DAS重建...')

    results = {}

    # 方法1: 单站模式 - 60度Hanning窗平滑
    print('\n--- 方法1: 单站(60°, Hanning, 平滑) ---')
    results['mono_60_hann'] = das_reconstruction_single_monostatic(
        processed_data,
        angle_threshold=60,
        apod_type='hanning',
        smooth_transition=True,
        dynamic_range_db=60
    )

    # 方法2: 单站模式 - 45度角度更小
    print('\n--- 方法2: 单站(45°, Hanning, 平滑) ---')
    results['mono_45_hann'] = das_reconstruction_single_monostatic(
        processed_data,
        angle_threshold=45,
        apod_type='hanning',
        smooth_transition=True,
        dynamic_range_db=60
    )

    # 方法3: 单站模式 - 线性加权
    print('\n--- 方法3: 单站(60°, Linear, 平滑) ---')
    results['mono_60_linear'] = das_reconstruction_single_monostatic(
        processed_data,
        angle_threshold=60,
        apod_type='linear',
        smooth_transition=True,
        dynamic_range_db=60
    )

    # 方法4: 全发全收（如果需要对比）
    print('\n--- 方法4: 全发全收(60°, Hanning, 平滑) ---')
    results['bistatic_60_hann'] = das_reconstruction_full_bistatic(
        processed_data,
        angle_threshold=60,
        apod_type='hanning',
        smooth_transition=True,
        dynamic_range_db=60
    )

    # 5. 增强和保存
    print('\n[步骤5] 图像增强和保存...')
    enhanced_results = {}
    for key, img in results.items():
        enhanced_results[key] = enhance_image(img, method='clahe')
        cv2.imwrite(f'das_{key}_enhanced.png', enhanced_results[key])

    # 6. 可视化对比
    print('\n[步骤6] 可视化对比结果...')

    titles = {
        'mono_60_hann': '单站: 60°+Hanning+平滑',
        'mono_45_hann': '单站: 45°+Hanning+平滑',
        'mono_60_linear': '单站: 60°+Linear+平滑',
        'bistatic_60_hann': '全发全收: 60°+Hanning+平滑',
    }

    visualize_comparison(results, titles, 'das_aperture_comparison.png')

    print('\n' + '=' * 70)
    print('处理完成！')
    print('=' * 70)


if __name__ == '__main__':
    main()