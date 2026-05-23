"""
动作识别推理脚本 — 输入 .bin 雷达数据，输出动作类别 + 置信度 + 可视化
带详细处理过程输出 + 文件选择对话框
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from scipy.fft import fft, fftshift
from scipy.signal import spectrogram
from numpy import hanning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob, re
import time
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox

# ============================================================
# 0. 解决中文字体显示问题
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
warnings.filterwarnings('ignore', category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 切换模型：改下面这行 =====
# MODEL = "md"       # 单特征微多普勒
MODEL = "rtdtat"     # RT+DT+AT 三通道融合
# ================================

_MODELS = {
    "md":    "resnet18_single_md.pth",
    "rtdtat": "resnet18_fused_rtdtat.pth",
}
MODEL_PATH = os.path.join(os.path.dirname(__file__), _MODELS[MODEL])
CLASS_NAMES = ['bend','diedao','jump','qianshuai','run','sit','stand','swing','walk']
CLASS_CN = {
    'bend':'弯腰', 'diedao':'跌倒', 'jump':'跳跃', 'qianshuai':'前摔',
    'run':'跑步', 'sit':'静坐', 'stand':'站立', 'swing':'摆臂', 'walk':'走路'
}

# ============================================================
# 1. 加载模型
# ============================================================

def select_file():
    """打开文件选择对话框，让用户选择.bin文件"""
    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 置顶对话框

    # 设置初始目录
    initial_dir = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"

    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="请选择雷达数据文件 (.bin)",
        initialdir=initial_dir,
        filetypes=[
            ("雷达数据文件", "*.bin"),
            ("所有文件", "*.*")
        ],
        parent=root
    )

    root.destroy()  # 关闭tkinter
    return file_path

def select_file_simple():
    """简化版文件选择（使用命令行交互）"""
    print("\n请选择输入方式：")
    print("  1. 输入文件路径")
    print("  2. 从预设测试文件中选择")
    print("  3. 自动选择随机测试文件")

    choice = input("请输入选项 (1/2/3，默认3): ").strip()

    if choice == '1':
        file_path = input("请输入.bin文件的完整路径: ").strip()
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None
        return file_path

    elif choice == '2':
        # 扫描所有可用的测试文件
        base = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"
        available_files = []

        print("\n扫描可用的测试文件...")
        for cls in CLASS_NAMES:
            for dist in ['0.8m', '3m']:
                src = os.path.join(base, f'{cls}_{dist}')
                if os.path.exists(src):
                    bins = sorted([f for f in os.listdir(src) if f.endswith('.bin') and f[0].isdigit()])
                    for bin_file in bins:
                        available_files.append((cls, dist, os.path.join(src, bin_file)))

        if not available_files:
            print("❌ 未找到任何测试文件")
            return None

        # 显示文件列表
        print(f"\n找到 {len(available_files)} 个测试文件：")
        for i, (cls, dist, path) in enumerate(available_files[:20]):  # 最多显示20个
            print(f"  {i+1:2d}. {CLASS_CN[cls]} ({dist}) - {os.path.basename(path)}")

        if len(available_files) > 20:
            print(f"  ... 还有 {len(available_files)-20} 个文件")

        while True:
            try:
                idx = input(f"\n请选择文件 (1-{len(available_files)}): ").strip()
                idx = int(idx) - 1
                if 0 <= idx < len(available_files):
                    cls, dist, file_path = available_files[idx]
                    print(f"已选择: {CLASS_CN[cls]} ({dist}) - {os.path.basename(file_path)}")
                    return file_path
                else:
                    print(f"请输入 1-{len(available_files)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")

    else:  # 默认选项3
        import random
        base = r"D:\downlowd_cloud\方向2-雷达数据demo\DATA"
        available_files = []

        for cls in CLASS_NAMES:
            for dist in ['0.8m', '3m']:
                src = os.path.join(base, f'{cls}_{dist}')
                if os.path.exists(src):
                    bins = sorted([f for f in os.listdir(src) if f.endswith('.bin') and f[0].isdigit()])
                    if bins:
                        available_files.append((cls, dist, os.path.join(src, bins[-1])))

        if available_files:
            cls, dist, file_path = random.choice(available_files)
            print(f"自动选择: {CLASS_CN[cls]} ({dist}) - {os.path.basename(file_path)}")
            return file_path
        else:
            print("❌ 未找到可用的测试文件")
            return None

print("\n[1/5] 加载深度学习模型...")
start_time = time.time()
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print(f"      ✓ 模型加载完成 (耗时: {time.time()-start_time:.2f}s)")
print(f"      ✓ 设备: {DEVICE}")
print(f"      ✓ 类别数: {len(CLASS_NAMES)}")
print(f"      ✓ 类别列表: {CLASS_NAMES}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================================
# 2. 信号处理（和训练时完全一致）
# ============================================================
def parse_logfile(logfile_path):
    print(f"      → 解析雷达配置文件...")
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

    print(f"         - 采样点数: {para['ADCSamples']}")
    print(f"         - Chirp数/帧: {para['ChirpNum']}")
    print(f"         - 总帧数: {para['FrameNum']}")
    print(f"         - 距离分辨率: {para['dr']*100:.2f} cm")
    print(f"         - 最大不模糊速度: {para['lambda']/4/para['Chirptime']/para['GroupNum']:.2f} m/s")
    return para

def read_raw_data(filename, para):
    print(f"      → 读取原始雷达数据...")
    frame_length = para['FrameNum']
    samples_per_frame = para['ADCSamples'] * para['ChirpNum'] * para['GroupNum'] * para['numRX']
    ncols = samples_per_frame * frame_length // 2
    raw = np.fromfile(filename, dtype=np.int16)
    file_size_mb = raw.nbytes / (1024 * 1024)
    print(f"         - 文件大小: {file_size_mb:.2f} MB")
    print(f"         - 数据点数: {len(raw):,}")

    adcData = raw.reshape(4, ncols, order='F')
    retVal = np.zeros((2, ncols), dtype=np.complex64)
    retVal[0, :] = adcData[0, :] + 1j * adcData[2, :]
    retVal[1, :] = adcData[1, :] + 1j * adcData[3, :]
    retVal = retVal.reshape(para['ADCSamples'], para['numRX'] * para['GroupNum'],
                             para['ChirpNum'], frame_length, order='F')
    print(f"         - 数据维度: {retVal.shape} (距离×天线×chirp×帧)")
    return retVal

def radar_to_fused_image(bin_path, logfile_path=None):
    """雷达原始数据 → 三通道融合谱图 → PIL Image"""
    print(f"\n[2/5] 雷达信号处理...")
    start_time = time.time()

    if logfile_path is None:
        logfile_path = glob.glob(os.path.join(os.path.dirname(bin_path), '*_LogFile.txt'))[0]
        print(f"      ✓ 找到配置文件: {os.path.basename(logfile_path)}")

    para = parse_logfile(logfile_path)
    adcData = read_raw_data(bin_path, para)
    n_range, n_rx, n_chirps, n_frames = adcData.shape
    target_bin = int(0.8 / para['dr'])
    range_win = hanning(n_range).reshape(-1, 1)
    v_max = para['lambda'] / 4 / para['Chirptime'] / para['GroupNum']

    print(f"      → 提取三特征图...")
    rt_map = np.zeros((n_range, n_frames))
    dt_map = np.zeros((n_chirps, n_frames))
    all_slow_time = []

    for f_idx in range(n_frames):
        if f_idx % 20 == 0:
            print(f"         处理帧: {f_idx+1}/{n_frames}")

        frame = adcData[:, :, :, f_idx]
        frame_rx0 = frame[:, 0, :]
        range_fft = fft(frame_rx0 * range_win, axis=0)
        rt_map[:, f_idx] = np.mean(np.abs(range_fft), axis=1)
        search_bins = range(max(0, target_bin-6), min(n_range, target_bin+7))
        best_bin = search_bins[np.argmax(np.mean(np.abs(range_fft[search_bins, :]), axis=1))]
        slow_time = range_fft[best_bin, :]
        dt_map[:, f_idx] = fftshift(abs(fft(slow_time * hanning(n_chirps))))
        all_slow_time.append(slow_time)

    print(f"      → 短时傅里叶变换 (STFT)...")
    signal = np.concatenate(all_slow_time)
    f, t, Sxx = spectrogram(signal, fs=1.0/para['Chirptime'],
                            window='hann', nperseg=256, noverlap=200, nfft=512,
                            return_onesided=False)
    Sxx = fftshift(Sxx, axes=0); f = fftshift(f)
    f_max = 2 * v_max / para['lambda']
    md_map = np.abs(Sxx[np.abs(f) <= f_max, :])

    rt_db = 10 * np.log10(rt_map + 1e-10)
    dt_db = 10 * np.log10(dt_map + 1e-10)
    md_db = 10 * np.log10(md_map + 1e-10)

    print(f"      → 渲染三通道融合图...")

    def render(data):
        d = np.clip((data - data.min()) / (data.max() - data.min() + 1e-10), 0, 1)
        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        ax.imshow(d, aspect='auto', origin='lower', cmap='jet')
        ax.axis('off')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        rgb = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return np.array(Image.fromarray(rgb).resize((224, 224)))

    ch_r = render(rt_db)
    ch_g = render(dt_db)
    ch_b = render(md_db)
    fused = np.stack([ch_r[:,:,0], ch_g[:,:,1], ch_b[:,:,2]], axis=-1)

    elapsed = time.time() - start_time
    print(f"      ✓ 信号处理完成 (耗时: {elapsed:.2f}s)")
    print(f"      ✓ 输出图像尺寸: {fused.shape}")

    return Image.fromarray(fused), (rt_db, dt_db, md_db), para

# ============================================================
# 3. 预测
# ============================================================
def predict(bin_path):
    print(f"\n{'='*60}")
    print(f"输入文件: {bin_path}")
    print(f"{'='*60}")

    # 检查文件是否存在
    if not os.path.exists(bin_path):
        print(f"❌ 错误: 文件不存在 - {bin_path}")
        return None, 0

    # 信号处理
    fused_img, (rt, dt, md), para = radar_to_fused_image(bin_path)

    # 模型推理
    print(f"\n[3/5] 深度学习模型推理...")
    start_time = time.time()
    tensor = transform(fused_img).unsqueeze(0).to(DEVICE)
    print(f"      ✓ 输入张量形状: {tensor.shape}")

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    elapsed = time.time() - start_time
    print(f"      ✓ 推理完成 (耗时: {elapsed:.3f}s)")

    # 取 Top-3
    top3_idx = np.argsort(probs)[::-1][:3]
    pred_class = CLASS_NAMES[top3_idx[0]]

    # 输出详细预测结果
    print(f"\n[4/5] 预测结果分析...")
    print(f"      Top-1: {CLASS_CN[CLASS_NAMES[top3_idx[0]]]} ({CLASS_NAMES[top3_idx[0]]}) - 置信度: {probs[top3_idx[0]]*100:.2f}%")
    print(f"      Top-2: {CLASS_CN[CLASS_NAMES[top3_idx[1]]]} ({CLASS_NAMES[top3_idx[1]]}) - 置信度: {probs[top3_idx[1]]*100:.2f}%")
    print(f"      Top-3: {CLASS_CN[CLASS_NAMES[top3_idx[2]]]} ({CLASS_NAMES[top3_idx[2]]}) - 置信度: {probs[top3_idx[2]]*100:.2f}%")

    # 计算预测置信度统计
    max_prob = probs[top3_idx[0]]
    second_prob = probs[top3_idx[1]]
    confidence_gap = (max_prob - second_prob) * 100
    print(f"      置信度差距: {confidence_gap:.2f}%")

    if max_prob > 0.9:
        print(f"      ✓ 预测结果可靠 (高置信度)")
    elif max_prob > 0.7:
        print(f"      ⚠ 预测结果可信度中等")
    else:
        print(f"      ⚠ 预测结果可信度较低，请检查输入数据")

    # ---- 可视化 ----
    print(f"\n[5/5] 生成可视化报告...")
    start_time = time.time()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, data, title in [
        (axes[0,0], rt, 'RT 距离-时间'),
        (axes[0,1], dt, 'DT 多普勒-时间'),
        (axes[0,2], md, '微多普勒'),
    ]:
        ax.imshow(data, aspect='auto', origin='lower', cmap='jet')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    axes[1,0].imshow(fused_img)
    axes[1,0].set_title('三通道融合输入', fontsize=11)
    axes[1,0].axis('off')

    ax = axes[1,1]
    ax.axis('off')
    colors = ['#2ecc71' if i==0 else '#3498db' if i==1 else '#95a5a6' for i in range(3)]
    bars = ax.barh(
        [CLASS_CN[CLASS_NAMES[i]] for i in top3_idx][::-1],
        [probs[i]*100 for i in top3_idx][::-1],
        color=colors[::-1]
    )
    for bar, (p, idx) in zip(bars, [(probs[i], i) for i in top3_idx][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{p*100:.1f}%', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.set_title('预测结果 (Top-3)', fontsize=12)

    axes[1,2].axis('off')
    info = (
        f"预测: {CLASS_CN[pred_class]} ({pred_class})\n"
        f"置信度: {probs[top3_idx[0]]*100:.1f}%\n\n"
        f"雷达参数:\n"
        f"  载频: {para['f0']/1e9:.0f} GHz\n"
        f"  距离分辨率: {para['dr']*100:.1f} cm\n"
        f"  帧数: {para['FrameNum']}\n"
        f"  总时长: {para['FrameNum']*para['ChirpNum']*para['Chirptime']:.1f}s"
    )
    axes[1,2].text(0, 0.5, info, transform=axes[1,2].transAxes,
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    axes[1,2].set_title('详情', fontsize=11)

    plt.suptitle(f'动作识别结果: {CLASS_CN[pred_class]}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"      ✓ 可视化完成 (耗时: {elapsed:.2f}s)")
    print(f"      ✓ 报告保存: prediction_result.png")

    return pred_class, probs[top3_idx[0]]

# ============================================================
# 4. 命令行入口
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("启动")
    print("=" * 60)

    total_start = time.time()

    # 如果命令行提供了参数，直接使用
    if len(sys.argv) >= 2:
        bin_file = sys.argv[1]
        print(f"使用命令行参数: {bin_file}")
        cls_pred, conf = predict(bin_file)
        if cls_pred:
            print(f"\n{'='*60}")
            print(f"最终结果")
            print(f"{'='*60}")
            print(f"  预测类别: {CLASS_CN[cls_pred]} ({cls_pred})")
            print(f"  置信度: {conf*100:.2f}%")
    else:
        # 没有参数，弹出文件选择对话框
        print("\n[文件选择] 请选择要识别的雷达数据文件...")

        # 尝试使用GUI文件选择器
        try:
            bin_file = select_file()
            if bin_file:
                print(f"已选择文件: {bin_file}")
                cls_pred, conf = predict(bin_file)
                if cls_pred:
                    print(f"\n{'='*60}")
                    print(f"最终结果")
                    print(f"{'='*60}")
                    print(f"  预测类别: {CLASS_CN[cls_pred]} ({cls_pred})")
                    print(f"  置信度: {conf*100:.2f}%")
            else:
                print("\n未选择任何文件，切换到命令行选择模式...")
                bin_file = select_file_simple()
                if bin_file:
                    cls_pred, conf = predict(bin_file)
                    if cls_pred:
                        print(f"\n{'='*60}")
                        print(f"最终结果")
                        print(f"{'='*60}")
                        print(f"  预测类别: {CLASS_CN[cls_pred]} ({cls_pred})")
                        print(f"  置信度: {conf*100:.2f}%")
                else:
                    print("❌ 未选择文件，程序退出")
                    sys.exit(1)
        except Exception as e:
            print(f"GUI文件选择器失败: {e}")
            print("切换到命令行选择模式...")
            bin_file = select_file_simple()
            if bin_file:
                cls_pred, conf = predict(bin_file)
                if cls_pred:
                    print(f"\n{'='*60}")
                    print(f"最终结果")
                    print(f"{'='*60}")
                    print(f"  预测类别: {CLASS_CN[cls_pred]} ({cls_pred})")
                    print(f"  置信度: {conf*100:.2f}%")
            else:
                print("❌ 未选择文件，程序退出")
                sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n总耗时: {total_elapsed:.2f}秒")
    print("结果图片已保存: prediction_result.png")
    print("\n" + "=" * 60)