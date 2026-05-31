# gen_dataset.py -- GPU-accelerated DT map dataset generation
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import json, os, gc, time
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ========== Load radar config ==========
script_dir = Path(__file__).parent
json_path = script_dir.parent / "4D" / "CCconfig_json" / "CCconfig_json.mmwave.json"
data_root = script_dir.parent / "datasets_4Dradar"

with open(json_path) as f:
    cfg = json.load(f)
dev = cfg["mmWaveDevices"][0]
prof = dev["rfConfig"]["rlProfiles"][0]["rlProfileCfg_t"]
frame_cfg = dev["rfConfig"]["rlFrameCfg_t"]

f0 = prof["startFreqConst_GHz"] * 1e9
lam = 3e8 / f0
fs = prof["digOutSampleRate"] * 1e3
slope = prof["freqSlopeConst_MHz_usec"] * 1e12
idle = prof["idleTimeConst_usec"] * 1e-6
ramp_end = prof["rampEndTime_usec"] * 1e-6
chirp_time = idle + ramp_end
bw = slope * prof["numAdcSamples"] / fs
dr = 3e8 / bw / 2

nSamples = prof["numAdcSamples"]
nChirps = frame_cfg["chirpEndIdx"] - frame_cfg["chirpStartIdx"] + 1
nLoops = frame_cfg["numLoops"]
nRX = 4
nChirpsPerFrame = nChirps * nLoops

r_min = int(0.5 / dr) + 1
r_max = int(5.0 / dr)
IMG_SIZE = 64

print("dr=%.3fm, r_min=%d, r_max=%d" % (dr, r_min, r_max))
print("nSamples=%d, nChirps=%d, nLoops=%d" % (nSamples, nChirps, nLoops))

# ========== DT map processing ==========
range_win = torch.hann_window(nSamples, device=DEVICE)
dopp_win = torch.hann_window(nLoops, device=DEVICE)

def process_scene(bin_path):
    with open(bin_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.int16)
    raw = raw.reshape(-1, 2)
    raw_c = raw[:, 0] + 1j * raw[:, 1]

    total_chirps = len(raw_c) // (nRX * nSamples)
    nFrames = total_chirps // nChirpsPerFrame
    if nFrames < 2:
        return None

    needed = nRX * nSamples * nFrames * nChirpsPerFrame
    raw_c = raw_c[:needed]
    raw_c = raw_c.reshape(nFrames, nChirpsPerFrame, nRX, nSamples)
    raw_c = raw_c.transpose(2, 3, 0, 1)
    raw_c = raw_c.reshape(nRX, nSamples, nFrames, nChirps, nLoops)
    data = raw_c.transpose(0, 1, 3, 4, 2)

    data_t = torch.from_numpy(data).to(DEVICE)
    del raw, raw_c, data
    gc.collect()

    # Range FFT dim 1
    rfft = torch.fft.fft(data_t * range_win[None, :, None, None, None], dim=1)
    # MTI dim 2
    rd_mti = rfft - rfft.mean(dim=2, keepdim=True)
    # Doppler FFT dim 3
    rd = torch.fft.fft(rd_mti * dopp_win[None, None, None, :, None], dim=3)
    rd = torch.fft.fftshift(rd, dim=3)
    # Power
    pwr = torch.mean(torch.sum(torch.abs(rd)**2, dim=2), dim=0)
    # DT map
    dt_map = torch.max(pwr[r_min-1:r_max, :, :], dim=0)[0].T

    del data_t, rfft, rd_mti, rd, pwr
    gc.collect()

    # Background norm
    bg = torch.quantile(dt_map, 0.1, dim=1, keepdim=True)
    dt_db = 10 * torch.log10(dt_map / (bg + 1e-6))
    dt_db = torch.clamp(dt_db, -30, 40)

    # Resize
    dt_db = dt_db.unsqueeze(0).unsqueeze(0)
    dt_resized = F.interpolate(dt_db, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    dt_resized = dt_resized.squeeze().cpu().numpy()

    del dt_map, bg, dt_db
    torch.cuda.empty_cache()
    return dt_resized.astype(np.float32)

# ========== Dataset generation ==========
class_names = ['bend','boxing','empty','falldown','jump',
               'liedown','run','sit','stand','swing','walk']
nClasses = len(class_names)
scenes_per_class = [100, 100, 20, 100, 100, 100, 100, 100, 100, 100, 100]
nTrain_per = [80, 80, 14, 80, 80, 80, 80, 80, 80, 80, 80]
nTest_per  = [20, 20, 6,  20, 20, 20, 20, 20, 20, 20, 20]

total_train = sum(nTrain_per)
total_test = sum(nTest_per)

X_train = np.zeros((total_train, IMG_SIZE, IMG_SIZE), dtype=np.float32)
y_train = np.zeros(total_train, dtype=np.int32)
X_test  = np.zeros((total_test, IMG_SIZE, IMG_SIZE), dtype=np.float32)
y_test  = np.zeros(total_test, dtype=np.int32)

train_idx, test_idx = 0, 0
print("=" * 60)
print("Generating: %d train + %d test = %d total" % (total_train, total_test, total_train+total_test))
print("=" * 60)

t0 = time.time()
total_done = 0

for cls, name in enumerate(class_names):
    nTotal = scenes_per_class[cls]
    nTr = nTrain_per[cls]
    nTe = nTest_per[cls]
    perm = np.random.permutation(nTotal)
    tr_scenes = perm[:nTr]
    te_scenes = perm[nTr:nTr+nTe]

    print("[%s] %d scenes, train=%d, test=%d" % (name, nTotal, nTr, nTe))

    for idx in tr_scenes:
        scenario = "CCdata_%s_%04d" % (name, idx+1)
        bin_path = data_root / scenario / "master_0000_data.bin"
        if not bin_path.exists():
            continue
        dt = process_scene(str(bin_path))
        if dt is None:
            continue
        X_train[train_idx] = dt
        y_train[train_idx] = cls
        train_idx += 1
        total_done += 1
        if total_done % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / total_done * (total_train + total_test - total_done)
            print("  Progress: %d/%d (%.1f%%) ETA: %.1f min" % (total_done, total_train+total_test, 100*total_done/(total_train+total_test), eta/60))

    for idx in te_scenes:
        scenario = "CCdata_%s_%04d" % (name, idx+1)
        bin_path = data_root / scenario / "master_0000_data.bin"
        if not bin_path.exists():
            continue
        dt = process_scene(str(bin_path))
        if dt is None:
            continue
        X_test[test_idx] = dt
        y_test[test_idx] = cls
        test_idx += 1
        total_done += 1

X_train = X_train[:train_idx]
y_train = y_train[:train_idx]
X_test  = X_test[:test_idx]
y_test  = y_test[:test_idx]

elapsed = time.time() - t0
print("\nTotal: train=%d, test=%d, time=%.1f min" % (train_idx, test_idx, elapsed/60))

# Save
sio.savemat(str(script_dir / "dt_dataset.mat"), {
    'X_train': X_train.transpose(1, 2, 0),
    'y_train': y_train[:, None],
    'X_test':  X_test.transpose(1, 2, 0),
    'y_test':  y_test[:, None],
    'class_names': np.array([[n] for n in class_names], dtype=object)
})
mat_size = os.path.getsize(script_dir / "dt_dataset.mat") / 1024 / 1024
print("Dataset saved: dt_dataset.mat (%.1f MB)" % mat_size)
