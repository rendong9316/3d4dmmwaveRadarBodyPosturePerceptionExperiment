"""
批量微多普勒谱图生成脚本
遍历 DATA/ 下的 .bin 文件，做 Range FFT → Doppler FFT → STFT → 保存 PNG
"""
import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import spectrogram
from numpy import hanning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time, glob, re
from PIL import Image

# ============================================================
# 1. 工具函数（复用 3d_processing.py 的逻辑）
# ============================================================

def parse_logfile(logfile_path):
    """解析DCA1000 LogFile，提取雷达参数"""
    para = {}
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'ProfileConfig' in line:
            a = [i for i, c in enumerate(line) if c == ',']
            para['IDLEtime'] = float(line[a[2]+1:a[3]]) * 1e-8
            para['STARTtime'] = float(line[a[3]+1:a[4]]) * 1e-8
            para['ENDtime'] = float(line[a[4]+1:a[5]]) * 1e-8
            para['FrequencySlope'] = float(line[a[7]+1:a[8]]) * 36210 / 1e6 * 1e12
            para['ADCSamples'] = int(float(line[a[9]+1:a[10]]))
            para['Fs'] = float(line[a[10]+1:a[11]]) * 1e3
            para['Chirptime'] = para['IDLEtime'] + para['ENDtime']
        if 'AdvancedFrameConfig' in line:
            a = [i for i, c in enumerate(line) if c == ',']
            para['GroupNum'] = int(float(line[a[4]+1:a[5]]))
            para['ChirpNum'] = int(float(line[a[5]+1:a[6]]))
            para['Frameinter'] = 0.5 * float(line[a[6]+1:a[7]]) * 1e-8
            para['FrameNum'] = int(float(line[a[38]+1:a[39]]))
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
    """读取DCA1000原始ADC数据"""
    if frame_length is None:
        frame_length = para['FrameNum'] - frame_start + 1
    samples_per_frame = para['ADCSamples'] * para['ChirpNum'] * para['GroupNum'] * para['numRX']
    ncols = samples_per_frame * frame_length // 2
    offset = (2 * samples_per_frame * (frame_start - 1)) * 2
    raw = np.fromfile(filename, dtype=np.int16, offset=offset)
    adcData = raw.reshape(4, ncols, order='F')
    retVal = np.zeros((2, ncols), dtype=np.complex64)
    retVal[0, :] = adcData[0, :] + 1j * adcData[2, :]
    retVal[1, :] = adcData[1, :] + 1j * adcData[3, :]
    retVal = retVal.reshape(para['ADCSamples'],
                             para['numRX'] * para['GroupNum'],
                             para['ChirpNum'],
                             frame_length, order='F')
    return retVal

def compute_microdoppler_spec(adcData, para, target_range_m=0.8):
    """
    从100帧原始数据生成微多普勒谱图
    返回: spectrogram [freq_bins, time_bins] 功率dB
    """
    n_range = para['ADCSamples']
    n_rx = para['numRX'] * para['GroupNum']
    n_chirps = para['ChirpNum']
    n_frames = adcData.shape[3]

    # 目标距离bin（0.8m附近）
    target_bin = int(target_range_m / para['dr'])

    # 对每帧做 Range FFT，提取目标距离bin的慢时间信号
    range_win = hanning(n_range).reshape(-1, 1)
    doppler_win = hanning(n_chirps)

    # 收集所有帧的目标距离bin信号
    all_slow_time = []

    for f_idx in range(n_frames):
        frame = adcData[:, :, :, f_idx]  # [256, 4, 245]
        # Range FFT (对RX0做)
        frame_rx0 = frame[:, 0, :]  # [256, 245]
        range_fft = fft(frame_rx0 * range_win, axis=0)
        # 取目标距离bin + 邻近4个bin的功率最强点
        search_bins = range(max(0, target_bin-4), min(n_range, target_bin+5))
        best_bin = search_bins[np.argmax(np.mean(np.abs(range_fft[search_bins, :]), axis=1))]
        # 提取该bin的慢时间信号
        slow_time = range_fft[best_bin, :]  # [245] complex
        all_slow_time.append(slow_time)

    # 拼接所有帧 → 长序列 [24500 samples, ~9.8秒]
    signal = np.concatenate(all_slow_time)

    # STFT
    nperseg = 256
    noverlap = 200
    nfft = 512
    f, t, Sxx = spectrogram(signal, fs=1.0/para['Chirptime'],
                            window='hann', nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft,
                            return_onesided=False)

    # fftshift 使零频居中，只取有意义的速度范围
    Sxx = fftshift(Sxx, axes=0)
    f = fftshift(f)

    # 限制频率范围到 ±v_max
    v_max = para['lambda'] / 4 / para['Chirptime'] / para['GroupNum']
    f_max = 2 * v_max / para['lambda']
    mask = np.abs(f) <= f_max
    Sxx = Sxx[mask, :]
    f = f[mask]

    # 转 dB
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)

    return Sxx_db, f, t

# ============================================================
# 2. 批量处理主函数
# ============================================================

def main():
    DATA_DIR = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"
    OUT_DIR  = r"D:\downlowd_cloud\方向2-雷达数据demo\training\dataset"

    # 选择要处理的类别和样本数
    TASKS = [
        ('stand_0.8m',    'stand',    120),
        ('stand_3m',      'stand',    120),
        ('swing_0.8m',    'swing',    120),
        ('swing_3m',      'swing',    120),
        ('jumpup_0.8m',   'jump',     120),
        ('jumpup_3m',     'jump',     120),
        ('bend_0.8m',     'bend',     120),
        ('bend_3m',       'bend',     120),
        ('sit_0.8m',      'sit',      120),
        ('sit_3m',        'sit',      120),
        ('run_0.8m',      'run',      120),
        ('run_3m',        'run',      120),
        ('walk',          'walk',     239),
        ('diedao_0.8m',   'diedao',   240),
        ('qianshuai_3m',  'qianshuai',240),
    ]

    # 一次性清空所有输出文件夹
    for cls_name in set(cls for _, cls, _ in TASKS):
        out_path = os.path.join(OUT_DIR, cls_name)
        os.makedirs(out_path, exist_ok=True)
        for f in os.listdir(out_path):
            os.remove(os.path.join(out_path, f))

    # 全局编号（同一类别不同距离连续编号）
    cls_counter = {}

    for src_folder, cls_name, max_samples in TASKS:
        src_path = os.path.join(DATA_DIR, src_folder)
        out_path = os.path.join(OUT_DIR, cls_name)
        os.makedirs(out_path, exist_ok=True)

        # 初始化该类的全局起始编号
        if cls_name not in cls_counter:
            cls_counter[cls_name] = len([f for f in os.listdir(out_path)
                                          if f.endswith('.png')])

        # 找 LogFile 解析参数（所有.bin共用一套参数）
        logfile = glob.glob(os.path.join(src_path, '*_LogFile.txt'))
        if not logfile:
            print(f"  [SKIP] {cls_name}: no LogFile found")
            continue
        para = parse_logfile(logfile[0])

        # 找所有 .bin 文件（排除 LogFile 附带的）
        bin_files = sorted(
            [f for f in os.listdir(src_path)
             if f.endswith('.bin') and f[0].isdigit()],
            key=lambda x: int(re.match(r'(\d+)', x).group(1))
        )[:max_samples]

        print(f"\n{'='*50}")
        print(f"处理: {cls_name} ({src_folder})")
        print(f"  参数: {len(bin_files)} 个 .bin, 距离分辨率 {para['dr']*100:.1f}cm")
        print(f"{'='*50}")

        t0 = time.time()
        count = 0
        for i, bf in enumerate(bin_files):
            bin_path = os.path.join(src_path, bf)
            try:
                adcData = read_raw_data(bin_path, para)
                Sxx_db, f, t = compute_microdoppler_spec(adcData, para)

                # 画图 → 保存（显式保存为RGB避免RGBA问题）
                fig, ax = plt.subplots(figsize=(2.24, 2.24))
                ax.imshow(Sxx_db, aspect='auto', origin='lower', cmap='jet')
                ax.axis('off')
                plt.tight_layout(pad=0)
                fig.canvas.draw()
                rgb = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
                plt.close(fig)
                png_path = os.path.join(out_path, f'{cls_name}_{cls_counter[cls_name]:03d}.png')
                Image.fromarray(rgb).resize((224, 224)).save(png_path)
                cls_counter[cls_name] += 1
                count += 1

                if (i+1) % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(bin_files)}] {elapsed:.0f}s, {count} 张完成")

            except Exception as e:
                print(f"  [ERROR] {bf}: {e}")

        elapsed = time.time() - t0
        print(f"  {cls_name} 完成: {count}/{len(bin_files)} 张, 耗时 {elapsed:.0f}s")

    print(f"\n{'='*50}")
    print("全量生成完成！")
    for cls in ['stand', 'swing', 'jump']:
        n = len(os.listdir(os.path.join(OUT_DIR, cls)))
        print(f"  {cls}: {n} 张")

if __name__ == '__main__':
    main()
