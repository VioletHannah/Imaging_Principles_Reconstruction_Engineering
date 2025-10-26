#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/14 15:46
# @Author : 箴澄
# @Func：
# @File : hw2_enhanced.py
# @Software: PyCharm
"""
CT 2D频域锥形滤波 (BPF) 实验
依赖 hw3.py 中的 parallel_projection / generate_sinogram
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from hw3 import load_image, generate_sinogram

def vectorized_back_projection(sinogram, thetas, img_shape):
    num_angles, num_detectors = sinogram.shape
    H, W = img_shape
    recon = np.zeros((H, W), dtype=np.float32)

    # 网格坐标 (以中心为原点)
    x = np.linspace(-(W - 1) / 2, (W - 1) / 2, W)
    y = np.linspace(-(H - 1) / 2, (H - 1) / 2, H)
    X, Y = np.meshgrid(x, y)

    s_coords = np.linspace(-(num_detectors - 1) / 2, (num_detectors - 1) / 2, num_detectors)
    cos_t = np.cos(np.deg2rad(thetas))
    sin_t = np.sin(np.deg2rad(thetas))

    for i, (c, s) in enumerate(zip(cos_t, sin_t)):
        s_pos = X * c - Y * s
        s_index = ((s_pos - s_coords[0]) / (s_coords[1] - s_coords[0])).astype(np.float32)
        s0 = np.floor(s_index).astype(int)
        s1 = s0 + 1
        valid = (s0 >= 0) & (s1 < num_detectors)
        w = s_index - s0
        proj_vals = np.zeros_like(s_pos, dtype=np.float32)
        proj_vals[valid] = (1 - w[valid]) * sinogram[i, s0[valid]] + w[valid] * sinogram[i, s1[valid]]
        recon += proj_vals

    recon -= recon.min()
    if recon.max() > 0:
        recon /= recon.max()
    return recon

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

def main(img_path = "phantom.bmp", out_dir = "cone_BPF_result"):

    os.makedirs(out_dir, exist_ok=True)

    # 读取图像
    img = load_image(img_path)
    img = cv2.resize(img, (128, 128))
    imgf = img.astype(np.float32) / 255.0

    # 生成 sinogram
    thetas = np.arange(0, 180)
    sinogram = generate_sinogram(imgf, thetas, width=1.0)
    cv2.imwrite(os.path.join(out_dir, "sinogram.png"), (sinogram / sinogram.max() * 255).astype(np.uint8))

    # 向量化反投影
    print("执行向量化反投影 ...")
    recon_bp = vectorized_back_projection(sinogram, thetas, imgf.shape)
    bp_mse = np.mean((recon_bp - imgf) ** 2)
    cv2.imwrite(os.path.join(out_dir, "recon_unfiltered.png"), (recon_bp * 255).astype(np.uint8))

    # 2D锥形滤波
    print("执行二维频域锥形滤波 ...")
    recon_bpf, filt = cone_filter_2d(recon_bp)
    mse_bpf = np.mean((recon_bpf - imgf) ** 2)
    print(f"滤波完成，MSE = {mse_bpf:.6e}")

    # 保存结果
    cv2.imwrite(os.path.join(out_dir, "recon_bpf.png"), (recon_bpf * 255).astype(np.uint8))
    plt.imsave(os.path.join(out_dir, "cone_filter.png"), filt, cmap="viridis")

    # 可视化对比
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(imgf, cmap="gray")
    axs[0].set_title("原始图像")
    axs[1].imshow(recon_bp, cmap="gray")
    axs[1].set_title(f"反投影结果（未滤波）\nMSE={bp_mse:.6e}")
    axs[2].imshow(recon_bpf, cmap="gray")
    axs[2].set_title(f"锥形滤波后图像\nMSE={mse_bpf:.6e}")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"), dpi=300)
    plt.show()

    print(f"结果已保存到 {out_dir}/")


if __name__ == "__main__":
    main()
