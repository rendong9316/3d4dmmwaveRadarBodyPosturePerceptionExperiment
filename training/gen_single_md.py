"""
单特征微多普勒谱图生成 — 算法完全匹配 micdopplertest_rd.m
两遍扫描：①最强bin搜索 ②多bin融合提取慢时间+STFT
"""
import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import spectrogram, detrend
from numpy import hanning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os, time, glob, re

DATA_DIR = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"
OUT_DIR  = r"D:\downlowd_cloud\方向2-雷达数据demo\training\dataset_single"

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

def microdoppler_matched(adcData, para, target_dist=0.8):
    """完全匹配 micdopplertest_rd.m 的微多普勒算法"""
    n_range, n_rx, n_chirps, n_frames = adcData.shape
    target_bin = int(round(target_dist / para['dr']))
    range_win = hanning(n_range).reshape(-1,1)
    v_max = para['lambda']/(4*para['Chirptime']*para['GroupNum'])

    # ===== 第一阶段：最强距离bin搜索 =====
    energy_acc = np.zeros(n_range)
    for f in range(n_frames):
        sig = adcData[:,0,:,f].copy()             # RX0 [n_range, n_chirps]
        sig = sig - sig.mean(axis=1, keepdims=True) # MTI
        rfft = fft(sig * range_win, axis=0)
        energy_acc += np.mean(np.abs(rfft)**2, axis=1)

    sr = max(1,target_bin-6), min(n_range,target_bin+7)
    best_bin = sr[0] + np.argmax(energy_acc[sr[0]:sr[1]])

    # ===== 第二阶段：多bin融合提取慢时间 =====
    bin_span = np.arange(-3, 4)
    all_slow = []
    for f in range(n_frames):
        sig = adcData[:,0,:,f].copy()
        sig = sig - sig.mean(axis=1, keepdims=True)  # MTI
        rfft = fft(sig * range_win, axis=0)

        bins_sel = best_bin + bin_span
        bins_sel = bins_sel[(bins_sel>=0) & (bins_sel<n_range)]
        slow = rfft[bins_sel,:].sum(axis=0)          # 多bin融合
        slow = slow.ravel()

        # 尖峰抑制
        amp = np.abs(slow)
        slow[amp > np.median(amp)*5] = 0
        all_slow.append(slow)

    signal = np.concatenate(all_slow)
    signal = detrend(signal)  # 去趋势

    # ===== STFT =====
    fs_chirp = 1.0 / para['Chirptime']
    F, T, S = spectrogram(signal, fs=fs_chirp, window='hann',
                           nperseg=256, noverlap=220, nfft=512,
                           mode='complex', return_onesided=False)
    S = fftshift(S, axes=0); F = fftshift(F)
    S_db = 20 * np.log10(np.abs(S) + 1e-6)

    # 速度轴截断
    f_max = 2*v_max / para['lambda']
    mask = np.abs(F) <= f_max
    return S_db[mask,:], F[mask], T, best_bin

def render_grayscale(data, size=(224,224)):
    """谱图 → jet渲染 → 灰度 → uint8"""
    d = np.clip((data - data.min())/(data.max()-data.min()+1e-10), 0, 1)
    fig, ax = plt.subplots(figsize=(2.24,2.24))
    ax.imshow(d, aspect='auto', origin='lower', cmap='jet')
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    rgb = np.array(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(fig)
    return np.array(Image.fromarray(rgb).resize(size).convert('L'))

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
                S_db, F, T, _ = microdoppler_matched(adc, para, dist)
                gray = render_grayscale(S_db)
                png = os.path.join(out, f'{cls_name}_{cnt[cls_name]:03d}.png')
                Image.fromarray(gray).save(png)
                cnt[cls_name] += 1; done += 1
                if done % 30 == 0:
                    print(f'  [{done}/{len(bins)}] {time.time()-t0:.0f}s')
            except Exception as e:
                print(f'  [ERR] {bf}: {e}')
        print(f'  Done: {done}, {time.time()-t0:.0f}s')

    print('\n===== 单特征数据集汇总 =====')
    for cls in sorted(cls_set):
        print(f'  {cls}: {len(os.listdir(os.path.join(OUT_DIR,cls)))}')

if __name__ == '__main__':
    main()
