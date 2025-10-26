#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/9/29 10:15
# @Author : 箴澄
# @Func：傅立叶反变换重建增强方法
# @File : hw1_enhanced.py
# @Software: PyCharm
"""
基于傅立叶反变换的CT图像重建增强实现
通过不同窗函数（Hamming、Hann）及无窗函数优化频域处理，降低重建误差
要求：
    1. 生成投影数据并显示正弦图
    2. 使用不同窗函数及无窗函数进行重建并显示结果
    3. 计算并比较不同方法的MSE
    4. 保存重建结果及评估数据
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.ndimage import rotate
from scipy.interpolate import griddata
from skimage.transform import resize


def calculate_mse(image1, image2):
    """
    计算两幅图像的均方误差(MSE)
    参数：
        image1: 第一幅图像（numpy数组）
        image2: 第二幅图像（numpy数组，需与第一幅尺寸相同）
    返回：
        mse: 均方误差值
    """
    if image1.shape != image2.shape:
        raise ValueError("输入图像尺寸必须一致")
    return np.mean((image1 - image2) **2)


def apply_filter(shape, type='hamming'):
    """
    生成二维窗函数
    参数：
        shape: 窗函数尺寸 (rows, cols)
        type: 窗函数类型，支持 'hamming'、'hann'、'none'
    返回：
        window_2d: 二维窗函数（numpy数组），'none'类型返回全1数组
    """
    rows, cols = shape
    if type == 'hamming':
        filter_row = np.hamming(rows)
        filter_col = np.hamming(cols)
    elif type == 'hann':
        filter_row = np.hanning(rows)
        filter_col = np.hanning(cols)
    elif type == 'none':
        # 不使用窗函数（全1数组，相当于不滤波）
        filter_row = np.ones(rows)
        filter_col = np.ones(cols)
    else:
        raise ValueError("不支持的窗函数类型，可选：'hamming'、'hann'、'none'")

    # 外积生成二维窗函数
    return np.outer(filter_row, filter_col)


def load_and_preprocess_image(img_path, target_size):
    """
    读取图像并预处理（转为灰度图、调整尺寸）
    参数：
        img_path: 图像路径
        target_size: 目标尺寸 (N, N)
    返回：
        img_gray: 预处理后的灰度图（归一化到0-1）
    """
    # 读取图像并转为灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_gray = np.array(img, dtype=np.float64)
    # 调整尺寸
    if img_gray.shape[0] != target_size or img_gray.shape[1] != target_size:
        img_gray = resize(img_gray, (target_size, target_size), order=3)
    # 归一化到0-1范围
    return img_gray / 255.0 if img_gray.max() > 1 else img_gray


def generate_sinogram(img, num_angles=180, angle_step=1):
    """
    生成正弦图（投影数据）
    参数：
        img: 输入图像（numpy数组）
        num_angles: 投影角度数量
        angle_step: 角度步长（度）
    返回：
        sinogram: 正弦图数据
        padded_size: 填充后的图像尺寸
        diag_len: 图像对角线长度（探测器尺寸）
    """
    p_N = img.shape[0]
    # 计算对角线长度（确保旋转后图像不被截断）
    diag_len = int(np.ceil(np.sqrt(2) * p_N))
    pad_amount = int(np.ceil((diag_len - p_N) / 2))
    pad_width = ((pad_amount, pad_amount), (pad_amount, pad_amount))

    # 图像填充
    img_padded = np.pad(img, pad_width, mode='constant', constant_values=0)
    padded_size = img_padded.shape[0]

    # 生成投影角度
    angles = np.arange(0, num_angles * angle_step, angle_step)
    sinogram = np.zeros((padded_size, num_angles), dtype=np.float64)

    # 逐角度生成投影
    for i, angle in enumerate(angles):
        # 旋转图像并计算投影（列方向求和）
        img_rotated = rotate(
            img_padded,
            angle,
            reshape=False,
            order=3,
            mode='constant',
            cval=0
        )
        sinogram[:, i] = np.sum(img_rotated, axis=0)

    # 裁剪到对角线长度
    center = padded_size // 2
    half_width = diag_len // 2
    sinogram = sinogram[center - half_width: center + half_width + (diag_len % 2), :]

    # 确保偶数行（便于FFT处理）
    if sinogram.shape[0] % 2 == 1:
        sinogram = np.vstack([sinogram, np.zeros((1, sinogram.shape[1]))])

    return sinogram, padded_size, diag_len


def frequency_domain_reconstruction(sinogram, padded_size, diag_len, target_size, filter_type):
    """
    基于频域的图像重建
    参数：
        sinogram: 正弦图数据
        padded_size: 填充后的尺寸
        diag_len: 对角线长度
        target_size: 目标输出尺寸
        filter_type: 窗函数类型
    返回：
        recon_img: 重建图像（归一化到0-1）
    """
    # 频域处理参数
    pad_N = 2048  # 频域填充尺寸
    pad_row_num = pad_N - sinogram.shape[0]
    pad_top = pad_row_num // 2
    pad_bottom = pad_row_num - pad_top

    # 正弦图填充并进行FFT
    sinogram_padded = np.pad(
        sinogram,
        ((pad_top, pad_bottom), (0, 0)),
        mode='constant'
    )
    freq_proj = np.fft.fftshift(
        np.fft.fft(
            np.fft.ifftshift(sinogram_padded, axes=0),
            axis=0
        ),
        axes=0
    )

    # 构建频率网格
    nfp = freq_proj.shape[0]
    omega_sino = np.arange(-(nfp - 1) / 2, (nfp - 1) / 2 + 1) * (2 * np.pi / nfp)
    angles_rad = np.arange(sinogram.shape[1]) * np.pi / 180  # 角度转为弧度
    theta_grid, omega_grid = np.meshgrid(angles_rad, omega_sino)

    # 频域极坐标到直角坐标转换
    omega_image = omega_sino
    omega_grid_x, omega_grid_y = np.meshgrid(omega_image, omega_image)
    coo_r_fft2 = np.sqrt(omega_grid_x** 2 + omega_grid_y **2)
    coo_th_fft2 = np.arctan2(omega_grid_y, omega_grid_x)
    coo_r_fft2 = coo_r_fft2 * np.sign(coo_th_fft2)
    coo_th_fft2[coo_th_fft2 < 0] += np.pi  # 统一角度范围到0-π

    # 极坐标插值到直角坐标
    points = np.vstack((theta_grid.ravel(), omega_grid.ravel())).T
    values = freq_proj.ravel()
    grid_points = np.vstack((coo_th_fft2.ravel(), coo_r_fft2.ravel())).T

    # 三次插值构建2D频域
    freq_2d = griddata(
        points,
        values,
        grid_points,
        method='cubic',
        fill_value=(0 + 0j)
    ).reshape(omega_grid_x.shape)

    # 应用窗函数
    window = apply_filter(freq_2d.shape, filter_type)
    freq_2d_windowed = freq_2d * window

    # 逆傅里叶变换得到重建图像
    recon_freq = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(freq_2d_windowed)))
    recon_img = np.abs(recon_freq)

    # 裁剪到目标尺寸
    crop_val = (pad_N - target_size) // 2
    if (pad_N - target_size) % 2 == 0:
        recon_img = recon_img[crop_val:-crop_val, crop_val:-crop_val]
    else:
        recon_img = recon_img[crop_val:crop_val + target_size, crop_val:crop_val + target_size]

    # 归一化
    return (recon_img - np.min(recon_img)) / (np.max(recon_img) - np.min(recon_img))


def plot_comparison(original_img, recon_imgs, mse_values, window_types, save_dir):
    """
    绘制并保存结果对比图
    参数：
        original_img: 原始图像
        recon_imgs: 重建图像列表
        mse_values: MSE值列表
        window_types: 窗函数类型列表
        save_dir: 保存目录
    """
    # 原始图像与重建结果对比（2x2布局适配3种重建结果）
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    for i, (recon_img, mse, w_type) in enumerate(zip(recon_imgs, mse_values, window_types)):
        plt.subplot(2, 2, i + 2)
        plt.imshow(recon_img, cmap='gray')
        plt.title(f'Reconstructed with {w_type}\nMSE: {mse:.6f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main(img_path='phantom.bmp', target_size=128, num_angles=180, angle_step=1, output_path='hw1_enhanced_results'):
    """
    主函数：执行增强版傅立叶反变换重建流程
    参数：
        img_path: 输入图像路径
        target_size: 重建图像尺寸
        num_angles: 投影角度数量
        angle_step: 角度步长（度）
        output_path: 结果保存目录
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 1. 图像读取与预处理
    print("读取并预处理图像...")
    original_img = load_and_preprocess_image(img_path, target_size)
    if original_img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    # 2. 生成正弦图
    print("生成投影数据（正弦图）...")
    global sinogram  # 用于绘图函数访问
    sinogram, padded_size, diag_len = generate_sinogram(
        original_img,
        num_angles=num_angles,
        angle_step=angle_step
    )
    snorm = sinogram / np.max(sinogram) * 255
    cv2.imwrite(os.path.join(output_path, 'sinogram.png'), snorm.astype(np.uint8))

    # 3. 不同窗函数重建
    print("开始图像重建...")
    filter_types = ['none', 'hann', 'hamming']
    recon_imgs = []
    mse_values = []

    for w_type in filter_types:
        print(f"使用{w_type}函数重建...")
        recon_img = frequency_domain_reconstruction(
            sinogram,
            padded_size,
            diag_len,
            target_size,
            w_type
        )
        recon_imgs.append(recon_img)

        # 计算MSE
        mse = calculate_mse(original_img, recon_img)
        mse_values.append(mse)
        print(f"{w_type}重建完成，MSE: {mse:.6f}")

        # 保存结果
        plt.imsave(os.path.join(output_path, f'reconstructed_{w_type}.png'), recon_img, cmap='gray')

    # 4. 结果可视化与保存
    print("绘制并保存结果对比图...")
    plot_comparison(original_img, recon_imgs, mse_values, filter_types, output_path)

    # 保存所有MSE结果
    with open(os.path.join(output_path, 'mse_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("不同窗函数重建MSE结果\n")
        f.write("=" * 30 + "\n")
        for w_type, mse in zip(filter_types, mse_values):
            f.write(f"{w_type}窗: {mse:.6f}\n")

    print("所有重建流程完成，结果已保存至:", output_path)

    return {
        'original_img': original_img,
        'sinogram': sinogram,
        'reconstructed_imgs': recon_imgs,
        'mse_values': mse_values,
        'filter_types': filter_types
    }


if __name__ == '__main__':
    results = main(
        img_path='phantom.bmp',
        target_size=128,
        num_angles=180,
        angle_step=1,
        output_path='hw1_enhanced_results'
    )