#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/9/29 20:20
# @Author : 箴澄
# @Func：反投影滤波法BPF
# @File : hw2.py
# @Software: PyCharm
"""
对附件中的图片进行平行投影，投影步进为1度（一个圆周为360度），一共投影180次。
采用滤波投影/投影滤波方法（学号为奇数的同学采用滤波投影法，学号为偶数的同学采用投影滤波法）进行数据重建。
要求：
    1. 将所有投影数据整合成灰度图像显示；
    2. 显示最终的重建图像；
    3. 计算重建图像和原始图像之间的MSE；
    4. 提出降低MSE的思路并进行尝试；
    5. 提交包含详细注释的全部源代码和报告（源代码以压缩包形式提交，报告以pdf文件提交）。
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from hw3 import load_image, generate_sinogram
from scipy.interpolate import interp1d

def backproject(sinogram, thetas, output_shape, width=1.0):
    """
    简单反投影函数，将sinogram反投影为图像
    参数:
        sinogram: 正弦图，形状为(num_angles, diag_length)
        thetas: 投影角度列表(度)，与sinogram的行对应
        output_shape: 输出图像的形状 (rows, cols)
        width: 条带宽，需与正投影时保持一致，默认1.0
    返回:
        recon: 反投影重建的图像
    """
    img_rows, img_cols = output_shape
    num_angles, diag_length = sinogram.shape

    # 初始化重建图像
    recon = np.zeros(output_shape, dtype=np.float64)

    # 计算像素中心相对于图像中心的坐标
    x_centers = np.arange(img_cols) - (img_cols - 1) / 2.0
    y_centers = np.arange(img_rows) - (img_rows - 1) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)  # 形状均为(img_rows, img_cols)

    # 生成投影轴s的坐标（与正投影保持一致）
    s_coords = np.linspace(-(diag_length - 1) / 2.0,
                          (diag_length - 1) / 2.0,
                          diag_length)
    s0 = s_coords[0]  # s坐标的起始值
    half = width / 2.0

    # 对每个角度进行反投影
    for i_theta in range(num_angles):
        theta_deg = thetas[i_theta]
        theta = np.deg2rad(theta_deg)

        # 计算该角度下每个像素的投影坐标s = x*cosθ - y*sinθ
        X_rot = X * np.cos(theta) - Y * np.sin(theta)

        # 计算每个像素对应的投影索引
        i_s = np.round(X_rot - s0).astype(int)
        # 确保索引在有效范围内
        i_s = np.clip(i_s, 0, diag_length - 1)

        # 获取当前角度的投影值并反投影到图像
        projection = sinogram[i_theta]
        recon += projection[i_s] # 等价于对每个像素做 recon[r,c] += projection[i_s[r,c]]

    return recon / recon.max()

def cone_filter_2d(img):
    """
    在二维频率域中进行锥形（ramp-like）滤波
    - 输入: 反投影得到的模糊图像 (float)
    - 输出: 滤波后图像 (float, 归一化)
    """
    # 转换到频域
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    H, W = img.shape
    u = np.linspace(-0.5, 0.5, W, endpoint=False)
    v = np.linspace(-0.5, 0.5, H, endpoint=False)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)

    # ---- 构造锥形滤波器 ----
    eps = 1e-6
    cone = D / (D.max() + eps)

    # 为防止低频信息完全丢失，可加入一个微小低频保留系数
    low_gain = 0.05
    H_filter = low_gain + (1 - low_gain) * cone

    # 应用滤波器
    F_filt = F_shift * H_filter
    f_ifft = np.fft.ifft2(np.fft.ifftshift(F_filt))
    f_real = np.real(f_ifft)

    # 归一化
    f_real -= f_real.min()
    if f_real.max() > 0:
        f_real /= f_real.max()

    return f_real, H_filter

def calculate_mse(original, reconstructed):
    """计算原始图像与重建图像的均方误差（MSE）"""
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    return np.mean((original - reconstructed) ** 2)

def main(image_path='input_image.png', img_size=128, num_projections = 180, out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)

    im = load_image(image_path)
    imf = im.astype(np.float32) / 255.0
    img = cv2.resize(imf, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rows, cols = img.shape
    print(f"原始图像尺寸：{rows}×{cols}")

    # 生成180次投影（0°~179°，步进1°）
    thetas = np.arange(0, num_projections, 1)  # 投影角度
    print("生成投影数据...")
    projections = generate_sinogram(img, thetas, width=1.0)
    # 保存正弦图
    sinogram_norm = (projections - projections.min()) / (projections.max() - projections.min())
    cv2.imwrite(os.path.join(out_dir, 'sinogram.png'), (sinogram_norm * 255).astype(np.uint8))

    print("正在重建图像...")
    recon_origin = backproject(projections, thetas, (rows, cols))
    cv2.imwrite(os.path.join(out_dir, 'recon_origin.png'), (recon_origin * 255).astype(np.uint8))
    mse_origin = calculate_mse(img, recon_origin)
    print(f"未滤波重建图像与原始图像的MSE: {mse_origin:.2f}")

    # 2D-FT滤波
    recon_processed, filter = cone_filter_2d(recon_origin)
    mse_ = calculate_mse(img, recon_processed)
    print(f"重建图像与原始图像的MSE: {mse_:.2f}")
    # 保存滤波器
    filter_norm = (filter - filter.min()) / (filter.max() - filter.min())
    cv2.imwrite(os.path.join(out_dir, 'cone_filter.png'), (filter_norm * 255).astype(np.uint8))

    # 尝试更多投影角度以降低MSE
    print("尝试增加投影角度数量以降低MSE...")
    extended_thetas = np.arange(0, 360, 1)  # 增加到360次投影
    extended_projections = generate_sinogram(img, extended_thetas, width=1.0)
    recon_extended = backproject(extended_projections, extended_thetas, (rows, cols))
    recon_extended_filtered, _ = cone_filter_2d(recon_extended)
    mse_extended = calculate_mse(img, recon_extended_filtered)
    print(f"增加投影角度后的重建图像与原始图像的MSE: {mse_extended:.2f}")


    # 可视化比较结果
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 15))
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    # 未滤波重建图像
    plt.subplot(2, 2, 2)
    plt.imshow(recon_origin, cmap='gray')
    plt.title(f'未滤波重建图像\nMSE: {mse_origin:.4f}')
    plt.axis('off')
    # 重建图像
    plt.subplot(2, 2, 3)
    plt.imshow(recon_processed, cmap='gray')
    plt.title(f'重建图像\nMSE: {mse_:.4f}')
    plt.axis('off')
    # 增加投影角度后的重建图像
    plt.subplot(2, 2, 4)
    plt.imshow(recon_extended_filtered, cmap='gray')
    plt.title(f'增加投影角度后的重建图像\nMSE: {mse_extended:.4f}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    main(image_path='phantom.bmp', out_dir='hw2_BPF')