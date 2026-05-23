"""
三通道融合谱图生成：R=RT, G=DT, B=AT（不含微多普勒）
算法匹配 micdopplertest_rd.m：MTI + 最强bin搜索 + 多bin融合 + 尖峰抑制
"""
import numpy as np
from scipy.fft import fft, fftshift
from numpy import hanning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os, time, glob, re

DATA_DIR = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"
OUT_DIR  = r"D:\downlowd_cloud\方向2-雷达数据demo\training\dataset_fused_rtdtat"

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

def parse_logfile(path):
    para = {}
    with open(path) as f:
        for line in f:
            if 'ProfileConfig' in line:
                a = [i for i,c in enumerate(line) if c==',']
                para['IDLEtime'] = float(line[a[2]+1:a[3]]) * 1e-8
                para['ENDtime']  = float(line[a[4]+1:a[5]]) * 1e-8
                para['FrequencySlope'] = float(line[a[7]+1:a[8]]) * 36210/1e6*1e12
                para['ADCSamples'] = int(float(line[a[9]+1:a[10]]))
                para['Fs'] = float(line[a[10]+1:a[11]]) * 1e3
                para['Chirptime'] = para['IDLEtime'] + para['ENDtime']
            if 'AdvancedFrameConfig' in line:
                a = [i for i,c in enumerate(line) if c==',']
                para['GroupNum'] = int(float(line[a[4]+1:a[5]]))
                para['ChirpNum']  = int(float(line[a[5]+1:a[6]]))
                para['Frameinter'] = 0.5*float(line[a[6]+1:a[7]])*1e-8
                para['FrameNum']  = int(float(line[a[38]+1:a[39]]))
    para['numRX']=4; para['f0']=60e9; para['lambda']=3e8/para['f0']
    para['BandWidth'] = para['FrequencySlope']*para['ADCSamples']/para['Fs']
    para['dr'] = 3e8/para['BandWidth']/2
    para['df'] = 1/para['Chirptime']/para['GroupNum']
    para['dv'] = para['lambda']*para['df']/2/para['ChirpNum']
    return para

def read_raw_data(filename, para):
    fl = para['FrameNum']
    spf = para['ADCSamples']*para['ChirpNum']*para['GroupNum']*para['numRX']
    ncols = spf*fl//2
    raw = np.fromfile(filename, dtype=np.int16)
    adc = raw.reshape(4, ncols, order='F')
    rv = np.zeros((2,ncols), dtype=np.complex64)
    rv[0,:]=adc[0,:]+1j*adc[2,:]; rv[1,:]=adc[1,:]+1j*adc[3,:]
    rv = rv.reshape(para['ADCSamples'], para['numRX']*para['GroupNum'],
                     para['ChirpNum'], fl, order='F')
    return rv

def extract_rt_dt_at(adcData, para, target_dist=0.8):
    """一次处理产出 RT + DT + AT 三张谱图"""
    n_range, n_rx, n_chirps, n_frames = adcData.shape
    target_bin = int(round(target_dist / para['dr']))
    range_win = hanning(n_range).reshape(-1,1)
    doppler_win = hanning(n_chirps)
    v_max = para['lambda']/(4*para['Chirptime']*para['GroupNum'])
    n_angle_fft = 128

    # ===== 第一阶段：最强距离bin搜索（同 micdopplertest_rd）=====
    energy_acc = np.zeros(n_range)
    for f in range(n_frames):
        sig = adcData[:,0,:,f].copy()
        sig = sig - sig.mean(axis=1, keepdims=True)  # MTI
        rfft = fft(sig * range_win, axis=0)
        energy_acc += np.mean(np.abs(rfft)**2, axis=1)

    sr = max(1,target_bin-6), min(n_range,target_bin+7)
    best_bin = sr[0] + np.argmax(energy_acc[sr[0]:sr[1]])

    # ===== 第二阶段：逐帧提取 RT, DT, AT =====
    rt_map = np.zeros((n_range, n_frames))
    dt_map = np.zeros((n_chirps, n_frames))
    at_map = np.zeros((n_angle_fft, n_frames))
    bin_span = np.arange(-3, 4)

    for f in range(n_frames):
        # --- 所有4通道做Range FFT ---
        frame_all_rx = adcData[:,:,:,f].copy()  # [n_range, 4, n_chirps]
        rfft_all = np.zeros_like(frame_all_rx, dtype=np.complex64)
        for rx in range(4):
            sig = frame_all_rx[:,rx,:]
            sig = sig - sig.mean(axis=1, keepdims=True)  # MTI
            rfft_all[:,rx,:] = fft(sig * range_win.ravel().reshape(-1,1), axis=0)

        # RX0 用于 RT/DT
        rfft_rx0 = rfft_all[:,0,:]  # [n_range, n_chirps]

        # --- RT 列：距离剖面 ---
        rt_map[:, f] = np.mean(np.abs(rfft_rx0)**2, axis=1)

        # --- 多bin融合提取慢时间 ---
        bins_sel = best_bin + bin_span
        bins_sel = bins_sel[(bins_sel>=0) & (bins_sel<n_range)]
        slow_sig = rfft_rx0[bins_sel,:].sum(axis=0)  # [n_chirps]
        amp = np.abs(slow_sig)
        slow_sig[amp > np.median(amp)*5] = 0  # 尖峰抑制

        # --- DT 列：多普勒 FFT ---
        dt = fftshift(abs(fft(slow_sig * doppler_win)))
        dt_map[:, f] = dt**2

        # --- AT 列：角度 FFT ---
        # 在最强距离bin处，取4个RX的复数值做角度FFT
        rx_vec = rfft_all[best_bin, :, :].sum(axis=1)  # [4] 跨chirp求和
        at_spec = fftshift(abs(fft(rx_vec, n_angle_fft)))
        at_map[:, f] = at_spec**2

    # 转 dB
    rt_db = 10 * np.log10(rt_map + 1e-10)
    dt_db = 10 * np.log10(dt_map + 1e-10)
    at_db = 10 * np.log10(at_map + 1e-10)

    return rt_db, dt_db, at_db, best_bin, v_max

def render_channel(data, size=(224,224)):
    """谱图 → jet渲染 → uint8 RGB"""
    d = np.clip((data - data.min())/(data.max()-data.min()+1e-10), 0, 1)
    fig, ax = plt.subplots(figsize=(2.24,2.24))
    ax.imshow(d, aspect='auto', origin='lower', cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    rgb = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(fig)
    return np.array(Image.fromarray(rgb).resize(size))

def main():
    # 清理
    cls_set = set(c for _,c,_ in CLASSES)
    for cls in cls_set:
        p = os.path.join(OUT_DIR, cls)
        os.makedirs(p, exist_ok=True)
        for f in os.listdir(p): os.remove(os.path.join(p,f))

    cnt = {}
    for src_folder, cls_name, max_n in CLASSES:
        src = os.path.join(DATA_DIR, src_folder)
        out = os.path.join(OUT_DIR, cls_name)
        if cls_name not in cnt:
            cnt[cls_name] = 0

        logs = glob.glob(os.path.join(src, '*_LogFile.txt'))
        if not logs: continue
        para = parse_logfile(logs[0])
        dist = 3.0 if '3m' in src_folder else 0.8

        bins = sorted([f for f in os.listdir(src)
                       if f.endswith('.bin') and f[0].isdigit()],
                      key=lambda x: int(re.match(r'(\d+)',x).group(1)))[:max_n]

        print(f'\n{cls_name} ({src_folder}): {len(bins)} files')
        t0 = time.time()
        done = 0
        for bf in bins:
            try:
                adc = read_raw_data(os.path.join(src,bf), para)
                rt, dt, at, _, _ = extract_rt_dt_at(adc, para, dist)

                # 三通道融合: R=RT, G=DT, B=AT
                ch_r = render_channel(rt)[:,:,0]
                ch_g = render_channel(dt)[:,:,1]
                ch_b = render_channel(at)[:,:,2]
                fused = np.stack([ch_r, ch_g, ch_b], axis=-1)

                png = os.path.join(out, f'{cls_name}_{cnt[cls_name]:03d}.png')
                Image.fromarray(fused).save(png)
                cnt[cls_name] += 1; done += 1
                if done % 30 == 0:
                    print(f'  [{done}/{len(bins)}] {time.time()-t0:.0f}s')
            except Exception as e:
                print(f'  [ERR] {bf}: {e}')
        print(f'  Done: {done}, {time.time()-t0:.0f}s')

    print('\n===== RT+DT+AT 融合数据集汇总 =====')
    for cls in sorted(cls_set):
        print(f'  {cls}: {len(os.listdir(os.path.join(OUT_DIR,cls)))}')

if __name__ == '__main__':
    main()
