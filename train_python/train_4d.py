
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, scipy.io as sio

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

ACTION_NAMES = ['walk','jump','swing','bend','boxing','sit','stand','run','falldown','liedown']
N_CLASSES = len(ACTION_NAMES)
X_RANGE = [-2.0, 2.0]; Y_RANGE = [0.5, 6.0]; Z_RANGE = [0.0, 2.5]
N_FRAMES = 15; IMG_SIZE = 64
BATCH_SIZE = 16; N_EPOCHS = 60; LR = 0.001

class RadarDataset(Dataset):
    def __init__(self, mat_path, split='train', train_ratio=0.7):
        print(f'Loading {mat_path}...')
        data = sio.loadmat(mat_path, simplify_cells=True)
        # Support both data formats
        if 'all_data' in data:
            d = data['all_data']
            self.scenarios = [str(s) for s in d['scenarios'].flatten()]
            self.labels = d['labels'].flatten().astype(np.int64)
            pc_raw = d['pointclouds'].flatten()
        else:
            self.scenarios = [str(s) for s in data['all_sc'].flatten()]
            self.labels = data['all_lb'].flatten().astype(np.int64)
            pc_raw = data['all_pc'].flatten()

        self.pointclouds = []
        for i in range(len(self.scenarios)):
            pts = np.atleast_2d(pc_raw[i])
            if pts.size == 0 or pts.shape[1] < 4:
                pts = np.zeros((1, 4))
            # Split into N_FRAMES pseudo-frames
            n_pts = pts.shape[0]
            pts_per_frame = max(1, n_pts // N_FRAMES)
            frames = []
            for f in range(N_FRAMES):
                start = f * pts_per_frame
                end = min(start + pts_per_frame, n_pts)
                chunk = pts[start:end]
                if len(chunk) == 0:
                    chunk = np.zeros((1, 4))
                frames.append(chunk)
            self.pointclouds.append(frames)

        idx = np.arange(len(self.scenarios))
        tr, te = train_test_split(idx, test_size=1-train_ratio, stratify=self.labels, random_state=42)
        self.indices = tr if split == 'train' else te
        print(f'  {split}: {len(self.indices)} samples')
    def __len__(self): return len(self.indices)
    def _project(self, pts):
        S = IMG_SIZE
        g = np.zeros((6, S, S), dtype=np.float32)
        c = np.zeros((3, S, S), dtype=np.float32)
        for p in pts:
            x,y,z,v = float(p[0]),float(p[1]),float(p[2]),float(p[3])
            ix=int((x-X_RANGE[0])/(X_RANGE[1]-X_RANGE[0])*S)
            iy=int((y-Y_RANGE[0])/(Y_RANGE[1]-Y_RANGE[0])*S)
            iz=int((z-Z_RANGE[0])/(Z_RANGE[1]-Z_RANGE[0])*S)
            if 0<=ix<S and 0<=iy<S: g[0,ix,iy]+=1; g[3,ix,iy]+=v; c[0,ix,iy]+=1
            if 0<=ix<S and 0<=iz<S: g[1,ix,iz]+=1; g[4,ix,iz]+=v; c[1,ix,iz]+=1
            if 0<=iy<S and 0<=iz<S: g[2,iy,iz]+=1; g[5,iy,iz]+=v; c[2,iy,iz]+=1
        for k in range(3):
            m=c[k]>0; g[k+3][m]/=c[k][m]; g[k]=np.clip(g[k],0,5)/5.0
        return g
    def __getitem__(self, idx):
        ri = self.indices[idx]
        proj = np.stack([self._project(f) for f in self.pointclouds[ri]], 0)
        return torch.FloatTensor(proj), torch.LongTensor([self.labels[ri]])[0]

class MultiViewCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_FRAMES*6, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3,2,1),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,n_classes))
    def forward(self, x):
        B,T,C,H,W=x.shape; x=x.view(B,T*C,H,W); x=self.net(x); x=x.view(B,-1); return self.fc(x)

def train_epoch(model, loader, opt, crit):
    model.train(); ls,cor,tot=0,0,0
    for d,t in loader:
        d,t=d.to(DEVICE),t.to(DEVICE); opt.zero_grad()
        out=model(d); loss=crit(out,t); loss.backward(); opt.step()
        ls+=loss.item(); cor+=(out.argmax(1)==t).sum().item(); tot+=t.size(0)
    return ls/len(loader), 100.*cor/tot

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); ls,cor,tot=0,0,0; preds,targs=[],[]
    for d,t in loader:
        d,t=d.to(DEVICE),t.to(DEVICE)
        out=model(d); ls+=crit(out,t).item()
        p=out.argmax(1); cor+=(p==t).sum().item(); tot+=t.size(0)
        preds.extend(p.cpu().tolist()); targs.extend(t.cpu().tolist())
    return ls/len(loader), 100.*cor/tot, preds, targs

def main():
    data_path = os.path.join(os.path.dirname(__file__), 'pointcloud_data', 'all_pointclouds.mat')
    ts = RadarDataset(data_path, 'train', 0.7)
    te = RadarDataset(data_path, 'test', 0.7)
    tl = DataLoader(ts, BATCH_SIZE, shuffle=True)
    vl = DataLoader(te, BATCH_SIZE, shuffle=False)
    print(f'Train: {len(ts)}, Test: {len(te)}')
    
    model = MultiViewCNN(N_CLASSES).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, 25, 0.5)
    
    best_acc=0; hist={'tr':[],'te':[]}
    for ep in range(N_EPOCHS):
        tl_l,tl_a=train_epoch(model,tl,opt,crit)
        vl_l,vl_a,preds,targs=evaluate(model,vl,crit)
        sch.step()
        hist['tr'].append(tl_a); hist['te'].append(vl_a)
        print(f'Epoch {ep+1:2d}: Tr={tl_a:.1f}% Te={vl_a:.1f}%')
        if vl_a>best_acc: best_acc=vl_a; torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__),'pointcloud_model.pth'))
    
    print()
    print(f"Best: {best_acc:.1f}%")
    
    # Confusion matrix
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),'best_model.pth'), map_location=DEVICE, weights_only=True))
    _,_,preds,targs=evaluate(model,vl,crit)
    cm=confusion_matrix(targs,preds)
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=ACTION_NAMES,yticklabels=ACTION_NAMES,ax=ax1,annot_kws={'size':7})
    ax1.set_title('Confusion Matrix'); ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    plt.setp(ax1.get_xticklabels(),rotation=45,ha='right',fontsize=7)
    ax2.plot(hist['tr'],'o-',ms=2,label='Train'); ax2.plot(hist['te'],'s-',ms=2,label='Test')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Training Curves')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__),'pointcloud_results.png'),dpi=150)
    print('Saved pointcloud_results.png')
    print()
    print(classification_report(targs,preds,target_names=ACTION_NAMES,zero_division=0))

if __name__=='__main__': main()
