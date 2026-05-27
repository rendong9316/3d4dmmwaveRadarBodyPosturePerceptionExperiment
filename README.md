# 毫米波雷达人体姿态感知实验

基于 TI 毫米波雷达（IWR6843 / AWR2243 级联）的人体运动姿态感知项目，涵盖原始 ADC 数据采集、信号处理流水线（Range-FFT → Doppler-FFT → CFAR → 角度估计 → 点云生成）及深度学习分类。

## 项目结构

```
方向2-雷达数据demo/
│
├── 3D雷达信号数据读取参考代码/    # TI DCA1000 原始数据读取
│   ├── readPara.m                 #   解析雷达配置参数
│   └── readRawData.m              #   读取 .bin 原始 ADC 数据
│
├── 3D/                            # 3D 雷达数据集（IWR6843, 60GHz）
│   ├── stand_0.8m/                #   静止站立，距离 0.8m
│   ├── jumpup_0.8m/               #   跳跃，距离 0.8m
│   ├── swing_0.8m/                #   摆动，距离 0.8m
│   └── white/                     #   噪声（无目标）
│
├── 4D/                            # 4D 级联雷达数据集（AWR2243, 77GHz）
│   ├── CCconfig_json/             #   雷达配置 JSON
│   ├── CCdata_stand_0001/         #   静止站立（1主+3从）
│   └── CCdata_empty_0002/         #   无目标（1主+3从）
│
├── 4Dproject/                     # 4D 雷达处理脚本
│   ├── read4DParam.m              #   4D 参数解析
│   ├── read4DRawData.m            #   4D 原始数据读取
│   ├── main_4d.m                  #   4D 主处理流水线
│   ├── micdopplertest_4d.m        #   微多普勒测试
│   └── pitch_spectrograms_4d.m    #   俯仰角谱图
│
├── training/                      # Python 深度学习训练
│   ├── gen_spectrograms.py        #   生成单特征微多普勒谱图
│   ├── gen_fused_spectrograms.py  #   生成多特征融合谱图
│   ├── gen_single_md.py           #   单特征微多普勒数据集
│   ├── gen_fused_rtdtat.py        #   RT+DT+AT 三通道融合数据集
│   ├── train.py                   #   ResNet18 训练脚本
│   └── infer.py                   #   推理脚本
│
├── 3d_processing.py               # 3D 数据处理（Python 版本）
│
│   # ↓ MATLAB 信号处理模块
├── rangeFFT.m                     # 距离维 FFT（快时间维，Hanning 窗）
├── dopplerFFT.m                   # 多普勒维 FFT（慢时间维，fftshift 零速居中）
├── staticClutterSuppression.m     # 静态杂波抑制（零速置零/MTI/相位均值相消）
├── cfar2D.m                       # 二维 CA-CFAR 目标检测
├── angleFFT.m                     # 角度 FFT 方位角估计（ULA 阵列）
├── generatePointCloud.m           # 检测点 → 3D 笛卡尔点云
├── visualizePointCloud.m          # 点云可视化
│
│   # ↓ MATLAB 演示/测试脚本
├── main_3d.m                      # 3D 入门脚本：RD 谱可视化（四场景对比）
├── main_3d_by_rendong_self.m      # 3D 完整流水线：读取→FFT→杂波抑制→CFAR→角度→点云
├── RTDTtest_rd.m                  # Range-Time / Doppler-Time 测试
├── micdopplertest_rd.m            # 微多普勒特征测试
└── rd_cfar_2d.m                   # RD 谱 + CFAR 检测可视化
```

## 雷达参数

### 3D 雷达（IWR6843）

| 参数 | 值 |
|------|-----|
| 载频 | 60 GHz |
| 接收通道 | 4 RX |
| ADC 采样数 | 256 |
| 采样率 | 10 MHz |
| 调频斜率 | 29.98 MHz/μs |
| 带宽 | 767.5 MHz |
| 距离分辨率 | 19.5 cm |
| 每帧 Chirp 数 | 245 |
| 帧数/场景 | 100 |
| 采集板 | DCA1000 |

### 4D 雷达（AWR2243 级联）

| 参数 | 值 |
|------|-----|
| 载频 | 77 GHz |
| 芯片数量 | 4 片（1 主 + 3 从） |
| 虚拟通道 | 12 个 |
| 距离分辨率 | 5 cm |
| 最大探测距离 | 10 m |
| 采集板 | TDA2XX |

## 快速开始

### 环境要求

**MATLAB（信号处理）**：R2023a 及以上，需 Signal Processing Toolbox。

**Python（深度学习）**：3.11+，PyTorch + torchvision。

### 3D 数据处理

1. 打开 MATLAB，将项目根目录设为工作路径
2. 运行 `main_3d.m` — 绘制四场景 Range-Doppler 谱图对比
3. 运行 `main_3d_by_rendong_self.m` — 完整流水线（含 CFAR 检测、角度估计、3D 点云）

```matlab
run('main_3d.m');
```

### 4D 数据处理

```matlab
cd 4Dproject
run('main_4d.m');
```

### 深度学习训练

```bash
cd training

# 1. 生成微多普勒谱图数据集
python gen_spectrograms.py

# 2. 训练单特征模型
python train.py md

# 3. 训练三通道融合模型
python train.py rtdtat

# 4. 推理
python infer.py
```

## 信号处理流水线

```
原始 ADC 数据 (.bin)
    │
    ▼
静态杂波抑制（帧间均值相消 / 零速通道置零 / MTI）
    │
    ▼
Range-FFT（256 点，Hanning 窗）→ 距离谱
    │
    ▼
Doppler-FFT（245 点，Hanning 窗）→ Range-Doppler 谱
    │
    ▼
2D CA-CFAR 检测 → 目标点列表
    │
    ▼
角度 FFT（4 RX，零填充至 128 点）→ 方位角估计
    │
    ▼
极坐标 → 笛卡尔坐标 → 3D 点云 (x, y, v)
```

## 深度学习分类

- **模型**：ResNet18（预训练权重）
- **单特征模式**：输入微多普勒谱图（灰度图）
- **融合模式**：RT + DT + AT 三通道伪彩色图
- **类别**：站立、跳跃、摆动 等人体运动姿态

## 实验场景说明

| 场景 | 描述 | 预期 RD 谱特征 |
|------|------|---------------|
| white | 无目标（噪声） | 全图低功率，无显著峰值 |
| stand_0.8m | 静止站立 | 0.8m 处强峰，能量集中在零速附近 |
| swing_0.8m | 身体摆动 | 0.8m 处目标峰，多普勒在 ±1~2 m/s 间周期摆动 |
| jumpup_0.8m | 原地跳跃 | 0.8m 处目标峰，大幅正负交替的多普勒频移 |
| CCdata_empty_0002 | 4D 空场景 | 级联系统背景噪声参考 |
| CCdata_stand_0001 | 4D 站立 | 级联系统静态目标参考 |

## Git 仓库

```
git@github.com:rendong9316/3d4dmmwaveRadarBodyPosturePerceptionExperiment.git
```
