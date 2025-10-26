import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.interpolate import griddata
import os
import csv

def calculate_mse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse

def create_2d_window(shape, window_type='hamming'):
    rows, cols = shape
    if window_type == 'hamming':
        window_1d_row = np.hamming(rows)
        window_1d_col = np.hamming(cols)
    elif window_type == 'hann':
        window_1d_row = np.hanning(rows)
        window_1d_col = np.hanning(cols)
    elif window_type == 'blackman':
        window_1d_row = np.blackman(rows)
        window_1d_col = np.blackman(cols)

    window_2d = np.outer(window_1d_row, window_1d_col)
    return window_2d


def main():
    img_path = 'phantom.bmp'
    img = plt.imread(img_path)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img = np.dot(img[..., :3], rgb_weights)
    P = img / 255
    p_N = img.shape[0]  # 图像尺寸 (N x N)
    theta_range = 180
    theta_step = 1
    theta_N = int(theta_range / theta_step)  # 投影角度数量
    # pad_N = np.power(2, np.ceil(np.log2(p_N * np.sqrt(2)))).astype(int)
    pad_N = 2048

    if P.shape[0] != p_N:
        from skimage.transform import resize
        P = resize(P, (p_N, p_N), order=3)

    angles = np.arange(theta_N)
    diag_len = int(np.ceil(np.sqrt(2) * p_N))
    pad_amount = int(np.ceil((diag_len - p_N) / 2))
    pad_width = ((pad_amount, pad_amount), (pad_amount, pad_amount))
    P_padded = np.pad(P, pad_width, mode='constant', constant_values=0)
    proj_sino = np.zeros((P_padded.shape[0], theta_N), dtype='float64')

    for i in range(theta_N):
        img_rotated = rotate(P_padded,
                             angles[i],
                             reshape=False,
                             order=3,
                             mode='constant',
                             cval=0)
        proj_sino[:, i] = np.sum(img_rotated, axis=0)

    center_pixel = P_padded.shape[0] // 2
    half_width = diag_len // 2
    proj_sino = proj_sino[center_pixel - half_width:center_pixel + half_width +
                          (diag_len % 2), :]
    if proj_sino.shape[0] % 2 == 1:
        proj_sino = np.vstack([proj_sino, np.zeros((1, proj_sino.shape[1]))])
    pad_row_num = pad_N - proj_sino.shape[0]
    pad_top = pad_row_num // 2
    pad_bottom = pad_row_num - pad_top
    proj_sino = np.pad(proj_sino, ((pad_top, pad_bottom), (0, 0)),
                       mode='constant')
    f_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(proj_sino, axes=0),
                                     axis=0),
                          axes=0)

    nfp = f_p.shape[0]
    omega_sino = np.arange(-(nfp - 1) / 2,
                           (nfp - 1) / 2 + 1) * (2 * np.pi / f_p.shape[0])
    theta = angles * np.pi / 180
    theta_grid, omega_grid = np.meshgrid(theta, omega_sino)

    omega_image = omega_sino
    omega_grid_x, omega_grid_y = np.meshgrid(omega_image, omega_image)

    coo_r_fft2 = np.sqrt(omega_grid_x**2 + omega_grid_y**2)
    coo_th_fft2 = np.arctan2(omega_grid_y, omega_grid_x)
    coo_r_fft2 = coo_r_fft2 * np.sign(coo_th_fft2)

    coo_th_fft2[coo_th_fft2 < 0] += np.pi

    points = np.vstack((theta_grid.ravel(), omega_grid.ravel())).T
    values = f_p.ravel()
    grid_points = np.vstack((coo_th_fft2.ravel(), coo_r_fft2.ravel())).T

    Fourier2_radial_flat = griddata(points,
                                    values,
                                    grid_points,
                                    method='cubic',
                                    fill_value=(0 + 0j))
    Fourier2_radial = Fourier2_radial_flat.reshape(omega_grid_x.shape)
    window_types = ['hamming', 'hann', 'blackman']
    mse_results = {}

    for window_type in window_types:
        print(f'正在应用{window_type}窗...')
        window = create_2d_window(Fourier2_radial.shape, window_type)
        Fourier2_radial_windowed = Fourier2_radial * window
        print(f'{window_type}窗应用完成！')

        # 使用加窗后的频域数据进行逆变换
        target = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(Fourier2_radial_windowed)))

        crop_val = (pad_N - p_N) // 2
        if (pad_N - p_N) % 2 == 0:
            target = target[crop_val:-crop_val, crop_val:-crop_val]
        else:
            target = target[crop_val:crop_val + p_N, crop_val:crop_val + p_N]

        I_a = np.abs(target)
        I_a = (I_a - np.min(I_a)) / (np.max(I_a) - np.min(I_a))

        mse_linear = calculate_mse(I_a, P)
        mse_results[window_type] = {'mse_linear': mse_linear}

        print(f'{window_type}窗 - 线性缩放重建图像 MSE: {mse_linear:.6f}')

        save_dir = f'./results/{window_type}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(8, 8))
        plt.imshow(I_a, cmap='gray')
        plt.title(f'Reconstructed Image ({window_type})')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'reconstructed_linear.png'),
                    dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 保存MSE结果为CSV文件
        mse_csv_path = os.path.join(save_dir, 'mse_results.csv')
        with open(mse_csv_path, mode='w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Window Type', 'MSE (Linear)'])
            writer.writerow([window_type, mse_linear])
        print(f'{window_type}窗的MSE结果已保存到: {mse_csv_path}')

    # 保存所有窗函数的MSE结果
    with open('./results/mse_results_all_windows.txt', 'w',
              encoding='utf-8') as f:
        f.write('不同窗函数的MSE结果\n')
        f.write('=' * 30 + '\n')
        for window_type, mse in mse_results.items():
            f.write(f'{window_type}窗:\n')
            f.write(f'  线性缩放 MSE: {mse["mse_linear"]:.6f}\n')
            f.write('\n')

    save_dir = './results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.imshow(proj_sino, cmap='gray', aspect='auto')
    plt.title('Projection')
    plt.xlabel('Angle ')
    plt.ylabel('Detector Position')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, 'sinogram.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(I_a, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'reconstructed_linear.png'),
                dpi=300,
                bbox_inches='tight')
    plt.close()

    with open(os.path.join(save_dir, 'mse_results.txt'), 'w',
              encoding='utf-8') as f:
        f.write('重建图像质量评估结果\n')
        f.write('=' * 30 + '\n')
        f.write(f'图像尺寸: {p_N}x{p_N}\n')
        f.write(f'投影角度数量: {theta_N}\n')
        f.write(f'线性缩放重建图像 MSE: {mse_linear:.6f}\n')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(I_a, cmap='gray')
    axes[0].set_title(f'Reconstructed Image\nMSE: {mse_linear:.6f}')
    axes[0].axis('off')

    axes[1].imshow(P, cmap='gray')
    axes[1].set_title(f'Original Image ({p_N}x{p_N})')
    axes[1].axis('off')

    diff_image = np.abs(I_a - P)
    axes[2].imshow(diff_image, cmap='hot')
    axes[2].set_title('Difference Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_with_mse.png'),
                dpi=300,
                bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
