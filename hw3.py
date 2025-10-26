#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/7 22:28
# @Author : 箴澄
# @Func：算数迭代方法
# @File : hw3.py
# @Software: PyCharm
"""
对图片进行平行投影，投影步进为9度（一个圆周为360度），一共投影20次。
采用ART/SART/SIRT方法进行数据重建。要求：
    1. 将所有投影数据整合成灰度图像显示；
    2. 显示最终的重建图像；
    3. 计算重建图像和原始图像之间的MSE，对比用傅里叶重建法/投影滤波法/滤波投影法得到的结果；
    4. 提出降低MSE的思路并进行尝试；
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_image(path):
    """读取图像并转为灰度图"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    return img

def parallel_projection(img, theta_deg, width=1.0):
    """
    近似平行投影
    - img: 2D 灰度图像（numpy array）
    - theta_deg: 投影角度，度
    - width: 条带宽（像素），默认 1.0，相当于 half=0.5（半宽）
    返回:
      projection: 1D array，长度 = diag_length (整型)
    """
    img_rows, img_cols = img.shape
    x_centers = np.arange(img_cols) - (img_cols - 1) / 2.0
    y_centers = np.arange(img_rows) - (img_rows - 1) / 2.0
    # X[r,c] 给出像素 (r,c) 的 x 坐标；Y[r,c] 给出像素 (r,c) 的 y 坐标
    X, Y = np.meshgrid(x_centers, y_centers)   # both shape (img_rows, img_cols)
    diag_length = int(np.ceil(np.hypot(img_rows, img_cols)))
    s_coords = np.linspace(-(diag_length - 1) / 2.0, (diag_length - 1) / 2.0, diag_length)
    theta = np.deg2rad(theta_deg)
    # X_rot 表示每个像素中心在投影轴上的坐标 s = x*cosθ - y*sinθ, cuz y 轴是向下的
    X_rot = X * np.cos(theta) - Y * np.sin(theta)

    half = width / 2.0
    projection = np.zeros_like(s_coords)
    for i, s in enumerate(s_coords):
        # mask 表示哪些像素的投影坐标落在 [s-half, s+half)
        mask = np.abs(X_rot - s) <= half
        # 累加被选像素的强度
        projection[i] = np.sum(img[mask])

    return projection

def generate_sinogram(img, thetas, width=1.0):
    """
    对一组角度生成 sinogram
    返回数组形状: (num_angles, diag_length)
    """
    projs = [parallel_projection(img, th, width=width) for th in thetas]
    sinogram = np.vstack(projs)  # 每行对应一个角度
    return sinogram

def reconstruct_ART(sinogram, thetas, img_size=64, lambda_=0.5, width=1.0, iterations=1):
    num_angles, diag_length = sinogram.shape
    recon = np.zeros((img_size, img_size), dtype=float)

    x_centers = np.arange(img_size) - (img_size - 1) / 2.0
    y_centers = np.arange(img_size) - (img_size - 1) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)
    s_coords = np.linspace(-(diag_length - 1) / 2.0, (diag_length - 1) / 2.0, diag_length)

    for it in range(iterations):
        for ai, theta in enumerate(thetas):
            theta_rad = np.deg2rad(theta)
            X_rot = X * np.cos(theta_rad) - Y * np.sin(theta_rad)
            for k, s in enumerate(s_coords):
                mask = np.abs(X_rot - s) <= (width / 2.0)
                if not np.any(mask):
                    continue
                proj_est = np.sum(recon[mask])
                diff = sinogram[ai, k] - proj_est
                count = np.count_nonzero(mask)
                recon[mask] += lambda_ * diff / float(count)
        recon = np.clip(recon, 0.0, None)
    return recon

def reconstruct_SART(sinogram, thetas, img_size=64, lambda_=0.5, width=1.0, iterations=5):
    """
    SART（Simultaneous ART, 逐角度内同时更新）
    近似实现：
      - 对每个角度 i：
          - 计算该角度下的所有射线估计 proj_est
          - 计算残差 diff
          - 构建一个角度级的 backproj（将每条射线的 diff 均分给相应的像素）
          - 将 backproj 乘以 lambda_ 并直接加到 recon 上（角度内一次性更新）
    """
    num_angles, diag_length = sinogram.shape
    recon = np.zeros((img_size, img_size), dtype=float)

    x_centers = np.arange(img_size) - (img_size - 1) / 2.0
    y_centers = np.arange(img_size) - (img_size - 1) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)

    s_coords = np.linspace(-(diag_length - 1) / 2.0, (diag_length - 1) / 2.0, diag_length)

    for it in range(iterations):
        for ai, theta in enumerate(thetas):
            theta_rad = np.deg2rad(theta)
            X_rot = X * np.cos(theta_rad) - Y * np.sin(theta_rad)

            # 先计算该角度下投影的估计值
            proj_est = np.zeros(diag_length, dtype=float)
            for k, s in enumerate(s_coords):
                mask = np.abs(X_rot - s) <= (width / 2.0)
                if np.any(mask):
                    proj_est[k] = np.sum(recon[mask])
                else:
                    proj_est[k] = 0.0

            diff = sinogram[ai] - proj_est

            # 构建该角度的 backprojection（把每条射线的 diff 均分给相交的像素）
            backproj = np.zeros_like(recon, dtype=float)
            for k, s in enumerate(s_coords):
                mask = np.abs(X_rot - s) <= (width / 2.0)
                if not np.any(mask):
                    continue
                count = np.count_nonzero(mask)
                backproj[mask] += diff[k] / float(count)

            # 将角度级 backproj 按 lambda_ 加入 recon（一次性更新）
            recon += lambda_ * backproj

        recon = np.clip(recon, 0.0, None)
    return recon

def reconstruct_SIRT(sinogram, thetas, img_size=64, lambda_=0.5, width=1.0, iterations=10):
    """
    SIRT（同时迭代重建）
    - 每次循环计算所有投影的残差并把回投的修正平均后一次性应用到图像上
    - 实现：累积所有角度的回投修正（按像素求和），最后除以角度数再乘以 lambda_
    """
    num_angles, diag_length = sinogram.shape
    recon = np.zeros((img_size, img_size), dtype=float)

    x_centers = np.arange(img_size) - (img_size - 1) / 2.0
    y_centers = np.arange(img_size) - (img_size - 1) / 2.0
    X, Y = np.meshgrid(x_centers, y_centers)

    s_coords = np.linspace(-(diag_length - 1) / 2.0, (diag_length - 1) / 2.0, diag_length)

    for it in range(iterations):
        update = np.zeros_like(recon)  # 累积所有角度带来的更新
        for ai, theta in enumerate(thetas):
            theta_rad = np.deg2rad(theta)
            X_rot = X * np.cos(theta_rad) - Y * np.sin(theta_rad)

            proj_est = np.zeros(diag_length, dtype=float)
            for k, s in enumerate(s_coords):
                mask = np.abs(X_rot - s) <= (width / 2.0)
                if np.any(mask):
                    proj_est[k] = np.sum(recon[mask])
                else:
                    proj_est[k] = 0.0

            diff = sinogram[ai] - proj_est

            # 将 diff 回投并累积到 update
            for k, s in enumerate(s_coords):
                mask = np.abs(X_rot - s) <= (width / 2.0)
                if not np.any(mask):
                    continue
                count = np.count_nonzero(mask)
                # 这里把 diff 均分到被穿过的像素上，然后累积到 update 上
                update[mask] += diff[k] / float(count)

        # 全部角度处理完后一次性更新 recon（均值化）
        recon += (lambda_ / float(len(thetas))) * update
        recon = np.clip(recon, 0.0, None)  # 非负性
    return recon

def compute_mse(img1, img2):
    return float(np.mean((img1 - img2) ** 2))

def main(input_path=None, img_size=128,
         num_projs=20, step_deg=9.0,
         width=1.0,
         lambda_ART=0.5, lambda_SART=0.5, lambda_SIRT=0.5,
         iter_ART=1, iter_SART=5, iter_SIRT=10):
    """
     - input_path: 输入图片路径
     - img_size: 重建图片大小（正方形）
     - num_projs, step_deg: 投影数量与步长（num_projs = 360/step_deg）
     - width: 投影条带宽度
     - lambda_*: 各算法默认超参数
     - iter_*: 各算法迭代次数
     - out_dir: 输出目录
    """
    out_dir = 'lambda_{}_iter_{}'.format(lambda_ART, iter_ART)
    os.makedirs(out_dir, exist_ok=True)

    # 读取并预处理输入图像
    im = load_image(input_path)
    imf = im.astype(np.float32) / 255.0
    img = cv2.resize(imf, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # 保存预处理后的原图（便于比较）
    cv2.imwrite(os.path.join(out_dir, 'orig_resized.png'), (img * 255).astype(np.uint8))

    # 生成sinogram
    thetas = np.arange(0.0, num_projs * step_deg, step_deg)
    sinogram = generate_sinogram(img, thetas, width=width)
    snorm = sinogram - sinogram.min()
    if snorm.max() > 0:
        snorm = snorm / snorm.max()
    cv2.imwrite(os.path.join(out_dir, 'sinogram.png'), (snorm * 255).astype(np.uint8))

    # ---------- ART ----------
    print("开始 ART 重建 ...")
    recon_art = reconstruct_ART(sinogram, thetas, img_size=img_size, lambda_=lambda_ART, width=width, iterations=iter_ART)
    mse_art = compute_mse(img, recon_art)
    print(f"ART 完成，MSE = {mse_art:.6e}")
    cv2.imwrite(os.path.join(out_dir, 'recon_ART.png'), (np.clip(recon_art,0,1) * 255).astype(np.uint8))

    # ---------- SART ----------
    print("开始 SART 重建...")
    recon_sart = reconstruct_SART(sinogram, thetas, img_size=img_size, lambda_=lambda_SART, width=width, iterations=iter_SART)
    mse_sart = compute_mse(img, recon_sart)
    print(f"SART 完成，MSE = {mse_sart:.6e}")
    cv2.imwrite(os.path.join(out_dir, 'recon_SART.png'), (np.clip(recon_sart,0,1) * 255).astype(np.uint8))

    # ---------- SIRT ----------
    print("开始 SIRT 重建...")
    recon_sirt = reconstruct_SIRT(sinogram, thetas, img_size=img_size, lambda_=lambda_SIRT, width=width, iterations=iter_SIRT)
    mse_sirt = compute_mse(img, recon_sirt)
    print(f"SIRT 完成，MSE = {mse_sirt:.6e}")
    cv2.imwrite(os.path.join(out_dir, 'recon_SIRT.png'), (np.clip(recon_sirt,0,1) * 255).astype(np.uint8))

    # 将 MSE 写入文本文件
    with open(os.path.join(out_dir, f'mse_results.txt'), 'w') as f:
        f.write(f"ART MSE: {mse_art:.6e}; lamda={lambda_ART}; iteration={iter_ART}\n")
        f.write(f"SART MSE: {mse_sart:.6e}; lamda={lambda_SART}; iteration={iter_SART}\n")
        f.write(f"SIRT MSE: {mse_sirt:.6e}; lamda={lambda_SIRT}; iteration={iter_SIRT}\n")

    print("所有重建完成，结果保存在目录:", out_dir)
    return {
        'sinogram': sinogram,
        'recon_ART': recon_art,
        'recon_SART': recon_sart,
        'recon_SIRT': recon_sirt,
        'mse': {'ART': mse_art, 'SART': mse_sart, 'SIRT': mse_sirt}
    }


if __name__ == '__main__':
    results = main(
        input_path='phantom.bmp',
        img_size=128,
        num_projs=20,
        step_deg=9.0,
        width=1.0,
        lambda_ART=0.6,
        lambda_SART=0.6,
        lambda_SIRT=0.6,
        iter_ART=100,
        iter_SART=100,
        iter_SIRT=100
    )