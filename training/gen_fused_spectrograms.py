"""
多特征融合谱图生成：微多普勒 + RT + DT → 三通道 PNG
复用 gen_spectrograms.py 的信号处理管线，增加 RT/DT 计算
"""
import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import spectrogram
from numpy import hanning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, time, glob, re

# ============================================================
# 1. 工具函数
# ============================================================

def parse_logfile(logfile_path):
    para = {}
    with open(logfile_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'ProfileConfig' in line:
            a = [i for i, c in enumerate(line) if c == ',']
            para['IDLEtime'] = float(line[a[2]+1:a[3]]) * 1e-8
            para['ENDtime'] = float(line[a[4]+1:a[5]]) * 1e-8
            para['FrequencySlope'] = float(line[a[7]+1:a[8]]) * 36210 / 1e6 * 1e12
            para['ADCSamples'] = int(float(line[a[9]+1:a[10]]))
            para['Fs'] = float(line[a[10]+1:a[11]]) * 1e3
            para['Chirptime'] = para['IDLEtime'] + para['ENDtime']
        if 'AdvancedFrameConfig' in line:
            a = [i for i, c in enumerate(line) if c == ',']
            para['GroupNum'] = int(float(line[a[4]+1:a[5]]))
            para['ChirpNum'] = int(float(line[a[5]+1:a[6]]))
            para['FrameNum'] = int(float(line[a[38]+1:a[39]]))
    para['numRX'] = 4; para['f0'] = 60e9; para['lambda'] = 3e8 / para['f0']
    para['BandWidth'] = para['FrequencySlope'] * para['ADCSamples'] / para['Fs']
    para['dr'] = 3e8 / para['BandWidth'] / 2
    para['df'] = 1 / para['Chirptime'] / para['GroupNum']
    para['dv'] = para['lambda'] * para['df'] / 2 / para['ChirpNum']
    return para

def read_raw_data(filename, para, frame_start=1, frame_length=None):
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
    retVal = retVal.reshape(para['ADCSamples'], para['numRX'] * para['GroupNum'],
                             para['ChirpNum'], frame_length, order='F')
    return retVal

def all_features(adcData, para, target_range_m=0.8):
    """一次处理产出微多普勒 + RT + DT 三张谱图"""
    n_range, n_rx, n_chirps, n_frames = adcData.shape
    target_bin = int(target_range_m / para['dr'])
    range_win = hanning(n_range).reshape(-1, 1)

    # 预分配
    rt_map = np.zeros((n_range, n_frames))
    dt_map = np.zeros((n_chirps, n_frames))
    all_slow_time = []

    v_max = para['lambda'] / 4 / para['Chirptime'] / para['GroupNum']

    for f_idx in range(n_frames):
        frame = adcData[:, :, :, f_idx]
        frame_rx0 = frame[:, 0, :]
        range_fft = fft(frame_rx0 * range_win, axis=0)

        # ---- RT: 距离剖面 ----
        rt_map[:, f_idx] = np.mean(np.abs(range_fft), axis=1)

        # ---- DT + 微多普勒: 找目标距离bin ----
        search_bins = range(max(0, target_bin-6), min(n_range, target_bin+7))
        best_bin = search_bins[np.argmax(np.mean(np.abs(range_fft[search_bins, :]), axis=1))]

        # DT: 取该bin的多普勒剖面（需先做Doppler FFT）
        slow_time = range_fft[best_bin, :]
        doppler_prof = fftshift(abs(fft(slow_time * hanning(n_chirps))))
        dt_map[:, f_idx] = doppler_prof

        # 微多普勒: 收集慢时间信号
        all_slow_time.append(slow_time)

    # ---- 微多普勒 STFT ----
    signal = np.concatenate(all_slow_time)
    f, t, Sxx = spectrogram(signal, fs=1.0/para['Chirptime'],
                            window='hann', nperseg=256, noverlap=200, nfft=512,
                            return_onesided=False)
    Sxx = fftshift(Sxx, axes=0); f = fftshift(f)
    f_max = 2 * v_max / para['lambda']
    mask = np.abs(f) <= f_max
    md_map = np.abs(Sxx[mask, :])

    # ---- RT/DT 转 dB ----
    rt_db = 10 * np.log10(rt_map + 1e-10)
    dt_db = 10 * np.log10(dt_map + 1e-10)
    md_db = 10 * np.log10(md_map + 1e-10)

    return md_db, rt_db, dt_db

def save_channel(data_2d, size=(224, 224)):
    """把二维谱图渲染成灰度图 → 返回 uint8 numpy [H,W,3]"""
    data_norm = np.clip((data_2d - data_2d.min()) / (data_2d.max() - data_2d.min() + 1e-10), 0, 1)
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    ax.imshow(data_norm, aspect='auto', origin='lower', cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    rgb = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return np.array(Image.fromarray(rgb).resize(size))

def main():
    DATA_DIR = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"
    OUT_DIR  = r"D:\downlowd_cloud\方向2-雷达数据demo\training\dataset_fused"

    CLASSES = [
        ('stand_0.8m','stand',120), ('stand_3m','stand',120),
        ('swing_0.8m','swing',120), ('swing_3m','swing',120),
        ('jumpup_0.8m','jump',120), ('jumpup_3m','jump',120),
        ('bend_0.8m','bend',120),   ('bend_3m','bend',120),
        ('sit_0.8m','sit',120),     ('sit_3m','sit',120),
        ('run_0.8m','run',120),     ('run_3m','run',120),
        ('walk','walk',239),
        ('diedao_0.8m','diedao',240),
        ('qianshuai_3m','qianshuai',240),
    ]

    # 清空输出
    class_names = set(c for _, c, _ in CLASSES)
    for cls in class_names:
        p = os.path.join(OUT_DIR, cls)
        os.makedirs(p, exist_ok=True)
        for f in os.listdir(p): os.remove(os.path.join(p, f))

    cls_counter = {}
    for src_folder, cls_name, max_samples in CLASSES:
        src_path = os.path.join(DATA_DIR, src_folder)
        out_path = os.path.join(OUT_DIR, cls_name)
        os.makedirs(out_path, exist_ok=True)
        if cls_name not in cls_counter:
            cls_counter[cls_name] = len([f for f in os.listdir(out_path) if f.endswith('.png')])

        logfile = glob.glob(os.path.join(src_path, '*_LogFile.txt'))
        if not logfile: continue
        para = parse_logfile(logfile[0])

        bin_files = sorted(
            [f for f in os.listdir(src_path) if f.endswith('.bin') and f[0].isdigit()],
            key=lambda x: int(re.match(r'(\d+)', x).group(1))
        )[:max_samples]

        print(f'\n{cls_name} ({src_folder}): {len(bin_files)} files')
        t0 = time.time()
        count = 0
        for i, bf in enumerate(bin_files):
            try:
                adcData = read_raw_data(os.path.join(src_path, bf), para)
                md, rt, dt = all_features(adcData, para)

                # 三通道融合: R=RT, G=DT, B=微多普勒
                ch_r = save_channel(rt)
                ch_g = save_channel(dt)
                ch_b = save_channel(md)
                fused = np.stack([ch_r[:,:,0], ch_g[:,:,1], ch_b[:,:,2]], axis=-1)

                png = os.path.join(out_path, f'{cls_name}_{cls_counter[cls_name]:03d}.png')
                Image.fromarray(fused).save(png)
                cls_counter[cls_name] += 1
                count += 1

                if (i+1) % 30 == 0:
                    print(f'  [{i+1}/{len(bin_files)}] {time.time()-t0:.0f}s, {count} done')
            except Exception as e:
                print(f'  [ERR] {bf}: {e}')

        print(f'  Done: {count}/{len(bin_files)}, {time.time()-t0:.0f}s')

    print('\n===== 融合数据集汇总 =====')
    for cls in sorted(class_names):
        n = len(os.listdir(os.path.join(OUT_DIR, cls)))
        print(f'  {cls}: {n}')

if __name__ == '__main__':
    main()
