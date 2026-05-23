import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

base = r'D:\downlowd_cloud\方向2-雷达数据demo\training\dataset'
np.random.seed(42)

for cls, pattern in [
    ('stand', 'flat'),
    ('swing', 'osc'),
    ('jump', 'burst')
]:
    outdir = os.path.join(base, cls)
    os.makedirs(outdir, exist_ok=True)

    # 清除旧文件
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))

    for i in range(30):
        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        img = np.zeros((100, 100))

        if pattern == 'flat':
            img[48:52, :] = np.random.rand(4, 100) * 0.5 + 0.3
        elif pattern == 'osc':
            for t in range(100):
                center = 50 + int(15 * np.sin(2 * np.pi * t / 20))
                img[max(0,center-3):min(100,center+3), t] = np.random.rand()*0.8 + 0.2
        else:
            for t in [15, 18, 45, 48, 75, 78]:
                img[30:70, t:t+3] = np.random.rand(40, 3) * 0.9 + 0.1

        img += np.random.rand(100, 100) * 0.05
        ax.imshow(img, cmap='jet', aspect='auto')
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(outdir, f'{cls}_{i:03d}.png'), dpi=100,
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f'{cls}: {len(os.listdir(outdir))} 张')

print('Done')
