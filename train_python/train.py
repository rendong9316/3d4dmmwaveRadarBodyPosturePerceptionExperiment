# train.py -- DTͼ��������ѵ��
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, time, os

np.random.seed(42); torch.manual_seed(42)

BATCH_SIZE = 32; EPOCHS = 50; LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(script_dir, "dt_images")

# ======================== Data ========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.25])
])

train_ds = datasets.ImageFolder(os.path.join(img_dir, "train"), transform=transform)
test_ds  = datasets.ImageFolder(os.path.join(img_dir, "test"),  transform=transform)
class_names = train_ds.classes

print("=" * 60)
print("DT image action classification")
print("Train: %d | Test: %d" % (len(train_ds), len(test_ds)))
print("Classes: %s" % class_names)
print("Device: %s" % DEVICE)
print("=" * 60)

for i, name in enumerate(class_names):
    tr = sum(1 for _, l in train_ds if l == i)
    te = sum(1 for _, l in test_ds if l == i)
    print("  %-10s  train=%3d  test=%3d" % (name, tr, te))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ======================== Model ========================
class DTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x); x = self.avgpool(x); return self.classifier(x)

# ======================== Training ========================
model = DTClassifier(len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("\n" + "=" * 60)
print("Training: Epochs=%d Batch=%d LR=%.4f" % (EPOCHS, BATCH_SIZE, LR))
print("Params: %d" % sum(p.numel() for p in model.parameters()))
print("=" * 60)

train_losses, test_accs = [], []
best_acc, best_state = 0, None
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    train_losses.append(avg_loss); scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            _, pred = torch.max(model(inputs), 1)
            total += labels.size(0); correct += (pred == labels).sum().item()

    acc = 100.0 * correct / total; test_accs.append(acc)
    if acc > best_acc: best_acc = acc; best_state = model.state_dict().copy()
    if epoch % 5 == 0 or epoch == 1:
        print("Epoch %3d/%d | Loss: %.4f | Acc: %5.2f%% | Best: %5.2f%%" % (epoch, EPOCHS, avg_loss, acc, best_acc))

t_train = time.time() - t_start
print("\nTraining done! Time: %.1fs | Best accuracy: %.2f%%" % (t_train, best_acc))

# ======================== Confusion Matrix ========================
model.load_state_dict(best_state); model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        _, pred = torch.max(model(inputs), 1)
        all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.numpy())

all_preds = np.array(all_preds); all_labels = np.array(all_labels)
cm = confusion_matrix(all_labels, all_preds)
acc_per_class = cm.diagonal() / cm.sum(axis=1) * 100
overall = 100 * cm.diagonal().sum() / cm.sum()

print("\n" + "=" * 60)
print("Confusion Matrix:")
print("-" * 70)
hdr = "%-12s" % "Actual\\Pred"
for n in class_names: hdr += "%6s" % n
hdr += "%6s  %7s" % ("Total", "Acc%")
print(hdr); print("-" * 70)
for i, name in enumerate(class_names):
    row = "%-12s" % name
    for j in range(len(class_names)): row += "%6d" % cm[i,j]
    row += "%6d  %6.1f%%" % (cm[i].sum(), acc_per_class[i])
    print(row)
print("-" * 70)
print("Overall: %d/%d = %.2f%%\n" % (cm.diagonal().sum(), cm.sum(), overall))
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# ======================== Save ========================
torch.save({"model_state_dict": best_state, "class_names": class_names, "test_acc": best_acc},
           os.path.join(script_dir, "dt_classifier.pth"))
print("Model saved: dt_classifier.pth")

# ======================== Visualizations ========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(train_losses, "b-", alpha=0.7)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss"); axes[0].grid(alpha=0.3)

axes[1].plot(test_accs, "r-", linewidth=2)
axes[1].axhline(y=best_acc, color="g", linestyle="--", label="Best: %.1f%%" % best_acc)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Test Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

im = axes[2].imshow(cm, cmap="Blues", aspect="auto")
axes[2].set_xticks(range(len(class_names)))
axes[2].set_yticks(range(len(class_names)))
axes[2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
axes[2].set_yticklabels(class_names, fontsize=8)
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix (Acc: %.1f%%)" % overall)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if cm[i, j] > 0:
            c = "white" if cm[i, j] > cm.max()/2 else "black"
            axes[2].text(j, i, cm[i, j], ha="center", va="center", color=c, fontsize=7)
plt.colorbar(im, ax=axes[2]); plt.tight_layout()
plt.savefig(os.path.join(script_dir, "training_results.png"), dpi=150, bbox_inches="tight")
print("Results saved: training_results.png")

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ecc71" if x >= 70 else "#f39c12" if x >= 50 else "#e74c3c" for x in acc_per_class]
bars = ax.bar(range(len(class_names)), acc_per_class, color=colors, edgecolor="white")
for bar, a in zip(bars, acc_per_class):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, "%.1f%%" % a,
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_ylabel("Accuracy (%)"); ax.set_title("Per-Class Accuracy (Overall: %.1f%%)" % overall)
ax.set_ylim(0, 105); ax.axhline(y=overall, color="blue", linestyle="--", label="Overall")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.savefig(os.path.join(script_dir, "per_class_accuracy.png"), dpi=150, bbox_inches="tight")
print("Per-class accuracy saved: per_class_accuracy.png")
plt.close("all")

print("\n" + "=" * 60 + "\nAll done!")