# 第二次作业 滤波反投影（BPF）实验报告

## 实验步骤

1. **原图预处理**  
   读取 `phantom.bmp`，灰度化并归一化到 \([0,1]\)，缩放为 \(128 * 128\)。

2. **投影生成**  
   以步进 **1°**，生成从 **0° 到 179° 共 180 次** 平行投影（sinogram）。  
   投影计算：对像素中心坐标 (x,y) 在角度 theta 下计算投影轴坐标  
   $$
   s = x * cos(theta) - y * sin(theta)
   $$
   在对应 s 位置累加像素值得到每一角度的投影。

3. **反投影重建（未经滤波）**  
   将 sinogram 在各角度反投影累加得到初步重建图 `recon_origin.png`。对像素索引使用最近邻截断（`np.round` + `np.clip`）。

4. **二维频域锥形滤波（Cone Filter）**  
   对反投影结果做 2D-FFT，构造锥形增益  
   $$
   H(u,v)=\text{low\_gain} + (1-\text{low\_gain})\frac{\sqrt{u^2+v^2}}{\max\sqrt{u^2+v^2}}
   $$
   乘以频谱再做逆变换得到 `recon_processed.png`，并保存滤波器图 `cone_filter.png`。

5. **角度扩展实验**  
   将投影角数扩展到 **360 次（0°–359°）**，生成扩展投影并重复反投影和滤波，比较 MSE 改善情况（结果保存在 `comparison.png`）。

---

## 实验结果

### 图片
正弦图（180 次）：`hw2_BPF/sinogram.png`  
![](./hw2_BPF/sinogram.png) 

滤波器频谱：`hw2_BPF/cone_filter.png`  
![](./hw2_BPF/cone_filter.png) 

投影数扩展后比较图：`hw2_BPF/comparison.png`
![comparison](./hw2_BPF/comparison.png)

### MSE
| 实验情形 | 投影角度数 | 滤波 | MSE |
|---------|------------:|:----:|-----:|
| 未滤波重建 | 180 | 否 | 0.1696 |
| 滤波后重建 | 180 | 是 | 0.0677 |
| 扩展角度重建 | 360 | 是 | 0.0671 |

---

## 主要结论

- 直接反投影（未经滤波）结果模糊，低频能量主导导致边缘不清。  

- 在重建结果上施加二维锥形滤波能显著增强高频分量，改善边缘和细节，降低 MSE。  
- 增加投影角度（从 180 → 360）进一步减少条纹伪影并改善重建精度。  
- 可以考虑在反投影时采用线性/双线性插值替代最近邻索引以降低抽样伪影；  


---

## 输出文件清单（`hw2_BPF/`）
- `sinogram.png` — 正弦图（180 投影）  
- `recon_origin.png` — 未滤波反投影结果  
- `cone_filter.png` — 锥形滤波器可视化（归一化）  
- `recon_processed.png` — 频域滤波后重建结果  
- `comparison.png` — 原图 / 未滤波 / 滤波 / 扩展角度对比图  

