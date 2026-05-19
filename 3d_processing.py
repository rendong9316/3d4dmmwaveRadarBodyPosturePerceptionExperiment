"""
3D毫米波雷达数据处理 — DCA1000 + IWR6843 (60GHz)
读取原始ADC数据 → Range-FFT → Doppler-FFT → 可视化RD谱
"""

import numpy as np
from scipy.fft import fft, fftshift
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3D')


def parse_logfile(logfile_path):
    """解析DCA1000 LogFile.txt，提取雷达参数，对应 readPara.m"""
    para = {}
    with open(logfile_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if 'ProfileConfig' in line:
            a = [i for i, c in enumerate(line) if c == ',']
            para['IDLEtime'] = float(line[a[2] + 1:a[3]]) * 1e-8
            para['STARTtime'] = float(line[a[3] + 1:a[4]]) * 1e-8
            para['ENDtime'] = float(line[a[4] + 1:a[5]]) * 1e-8
            para['FrequencySlope'] = float(line[a[7] + 1:a[8]]) * 36210 / 1e6 * 1e12
            para['ADCSamples'] = int(float(line[a[9] + 1:a[10]]))
            para['Fs'] = float(line[a[10] + 1:a[11]]) * 1e3
            para['Chirptime'] = para['IDLEtime'] + para['ENDtime']

        if 'AdvancedFrameConfig' in line:
            a = [i for i, c in enumerate(line) if c == ',']
            para['GroupNum'] = int(float(line[a[4] + 1:a[5]]))
            para['ChirpNum'] = int(float(line[a[5] + 1:a[6]]))
            para['Frameinter'] = 0.5 * float(line[a[6] + 1:a[7]]) * 1e-8
            para['FrameNum'] = int(float(line[a[38] + 1:a[39]]))

    para['numRX'] = 4
    para['f0'] = 60e9
    para['lambda'] = 3e8 / para['f0']
    para['d'] = para['lambda'] / 2
    para['BandWidth'] = para['FrequencySlope'] * para['ADCSamples'] / para['Fs']
    para['dr'] = 3e8 / para['BandWidth'] / 2
    para['df'] = 1 / para['Chirptime'] / para['GroupNum']
    para['dv'] = para['lambda'] * para['df'] / 2 / para['ChirpNum']

    return para


def read_raw_data(filename, para, frame_start=1, frame_length=None):
    """
    读取DCA1000原始ADC数据，对应 readRawData.m
    返回: adcData [ADCSamples, numRX*GroupNum, ChirpNum, FrameLength] complex
    """
    if frame_length is None:
        frame_length = para['FrameNum'] - frame_start + 1

    samples_per_frame = para['ADCSamples'] * para['ChirpNum'] * para['GroupNum'] * para['numRX']
    ncols = samples_per_frame * frame_length // 2
    offset = (2 * samples_per_frame * (frame_start - 1)) * 2  # bytes, 每个int16占2字节

    raw = np.fromfile(filename, dtype=np.int16, offset=offset)
    adcData = raw.reshape(4, ncols, order='F')

    retVal = np.zeros((2, ncols), dtype=np.complex64)
    retVal[0, :] = adcData[0, :] + 1j * adcData[2, :]
    retVal[1, :] = adcData[1, :] + 1j * adcData[3, :]

    retVal = retVal.reshape(
        para['ADCSamples'],
        para['numRX'] * para['GroupNum'],
        para['ChirpNum'],
        frame_length,
        order='F'
    )
    return retVal


def compute_rd_map(adcData):
    """
    计算单帧 Range-Doppler 图
    adcData: [ADCSamples, numRX, ChirpNum] — 单个frame
    返回: RD_map [ChirpNum_fft, ADCSamples_fft] 功率(dB)
    """
    n_range = adcData.shape[0]
    n_chirps = adcData.shape[2]

    # 加窗
    range_win = np.hanning(n_range).reshape(-1, 1)
    doppler_win = np.hanning(n_chirps)

    # 对所有RX取平均（或可只用单RX）
    data = np.mean(adcData, axis=1)  # [ADCSamples, ChirpNum]

    # Range FFT
    range_fft = fft(data * range_win, axis=0)

    # Doppler FFT
    rd = fft(range_fft * doppler_win, axis=1)
    rd = fftshift(rd, axes=1)  # 零多普勒居中

    rd_db = 20 * np.log10(np.abs(rd) + 1e-10)
    return rd_db


def process_scenario(name, para):
    """处理单个场景，返回所有帧的RD图堆叠"""
    bin_file = f"{BASE}\\{name}\\1.bin"
    print(f"  Reading {bin_file} ...")
    adcData = read_raw_data(bin_file, para)
    n_frames = adcData.shape[3]
    print(f"  Shape: {adcData.shape}, Frames: {n_frames}")

    all_rd = []
    for f_idx in range(n_frames):
        frame = adcData[:, :, :, f_idx]
        rd = compute_rd_map(frame)
        all_rd.append(rd)

    return np.stack(all_rd, axis=2), n_frames


def main():
    # 1. 解析参数（所有3D场景用同一配置）
    logfile = f"{BASE}\\stand_0.8m\\1_LogFile.txt"
    para = parse_logfile(logfile)
    print("=" * 60)
    print("雷达参数".center(56))
    print("=" * 60)
    print(f"  载频:            {para['f0'] / 1e9} GHz")
    print(f"  ADC采样数:       {para['ADCSamples']}")
    print(f"  采样率:          {para['Fs'] / 1e3:.0f} kHz")
    print(f"  调频斜率:        {para['FrequencySlope'] / 1e12:.2f} MHz/us")
    print(f"  每帧Chirp数:     {para['ChirpNum']}")
    print(f"  帧数:            {para['FrameNum']}")
    print(f"  带宽:            {para['BandWidth'] / 1e6:.1f} MHz")
    print(f"  距离分辨率:      {para['dr'] * 100:.2f} cm")
    print(f"  速度分辨率:      {para['dv'] * 100:.2f} cm/s")
    print(f"  最大距离:        {3e8 * para['Fs'] / 2 / para['FrequencySlope']:.1f} m")
    print(f"  最大速度:        {para['lambda'] / 4 / para['Chirptime'] / para['GroupNum']:.2f} m/s")
    print("=" * 60)

    # 2. 处理所有场景
    scenarios = ['white', 'stand_0.8m', 'swing_0.8m', 'jumpup_0.8m']
    labels_cn = ['噪声（无目标）', '静止站立@0.8m', '摆动@0.8m', '跳跃@0.8m']
    results = {}

    for name in scenarios:
        print(f"\n[处理] {name}")
        results[name] = process_scenario(name, para)

    # 3. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # 计算坐标轴
    n_range = para['ADCSamples']
    n_doppler = para['ChirpNum']
    range_axis = np.arange(n_range) * para['dr']
    v_max = para['lambda'] / 4 / para['Chirptime'] / para['GroupNum']
    vel_axis = np.linspace(-v_max, v_max, n_doppler)

    for idx, (name, label) in enumerate(zip(scenarios, labels_cn)):
        all_rd, _ = results[name]
        # 取中间帧（目标最稳定）
        mid = all_rd.shape[2] // 2
        rd_db = all_rd[:, :, mid]

        im = axes[idx].imshow(rd_db, aspect='auto', origin='lower',
                               extent=[vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]],
                               cmap='jet', vmin=-40, vmax=40)
        axes[idx].set_title(label, fontsize=12)
        axes[idx].set_xlabel('速度 (m/s)')
        axes[idx].set_ylabel('距离 (m)')
        axes[idx].set_xlim([-3, 3])
        axes[idx].set_ylim([0, 3])

    # 统一 colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('功率 (dB)', rotation=270, labelpad=15)

    fig.suptitle('Range-Doppler 谱图对比（中间帧）', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{BASE}/../RD_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\n图表已保存至: {BASE}/../RD_comparison.png")

    # 4. 时域多普勒演变（以jumpup为例，沿帧看多普勒变化）
    all_rd_jump, n_frames = results['jumpup_0.8m']
    doppler_sum = np.sum(np.abs(all_rd_jump[30:80, :, :]), axis=0)  # 近距0.3~0.8m累加

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 多普勒-时间演变图
    ax1.imshow(doppler_sum.T, aspect='auto', origin='lower',
               extent=[0, n_frames - 1, vel_axis[0], vel_axis[-1]],
               cmap='jet')
    ax1.set_title('跳跃场景 — 多普勒-时间演变', fontsize=12)
    ax1.set_xlabel('帧序号')
    ax1.set_ylabel('速度 (m/s)')
    ax1.set_ylim([-3, 3])

    # 平均多普勒谱对比
    for name, label, color, ls in zip(
        scenarios, labels_cn,
        ['gray', 'blue', 'green', 'red'],
        ['-', '-', '-', '-']
    ):
        all_rd, _ = results[name]
        doppler_profile = np.mean(np.abs(all_rd[30:80, :, :]), axis=(0, 2))
        doppler_profile_db = 20 * np.log10(doppler_profile + 1e-10)
        ax2.plot(vel_axis, doppler_profile_db, label=label, color=color, linewidth=1.5)

    ax2.set_title('平均多普勒谱对比 (0.3~0.8m)', fontsize=12)
    ax2.set_xlabel('速度 (m/s)')
    ax2.set_ylabel('功率 (dB)')
    ax2.set_xlim([-3, 3])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{BASE}/../Doppler_evolution.png', dpi=200, bbox_inches='tight')
    plt.show()
    print(f"图表已保存至: {BASE}/../Doppler_evolution.png")


if __name__ == '__main__':
    main()
