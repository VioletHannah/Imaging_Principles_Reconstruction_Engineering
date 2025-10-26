#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/25 15:08
# @Author : 箴澄
# @Func：
# @Site : 
# @File : temp.py
# @Software: PyCharm

import scipy.io
import scipy.signal
import h5py
import numpy as np

def _read_h5(obj):
    # 递归读取 h5py 对象为 Python 原生类型或 numpy 数组
    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[()]
            # 将 bytes -> str（MATLAB 字符串）
            if data.dtype.kind == 'S':
                return data.astype(str)
            return data
        except Exception:
            return None
    elif isinstance(obj, h5py.Group):
        out = {}
        for key in obj:
            out[key] = _read_h5(obj[key])
        return out
    else:
        return None

def load_mat(path):
    """
    尝试用 scipy 加载；若为 v7.3（HDF5）则使用 h5py 返回字典。
    返回 dict: 变量名 -> numpy/嵌套 dict
    """
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        out = {}
        with h5py.File(path, 'r') as f:
            # 列出顶层 keys
            for key in f.keys():
                out[key] = _read_h5(f[key])
        return out

def main():
    path = r"C:\Users\LENOVO\OneDrive\桌面\code\成像原理与图像工程\B-mode.mat"
    data = load_mat(path)
    # 打印顶层变量名
    print("keys:", list(data.keys()))
    # 访问某个变量示例（替换为实际变量名）
    print(data['b'])
    # 把 data['b'] 写入npy文件
    np.save('b_data.npy', data['b'])
    print("Data 'b' saved to 'b_data.npy'")

def test_load_npy():
    test = np.load('b_data.npy')
    print("Loaded data shape:", test.shape)

if __name__ == '__main__':
    test_load_npy()