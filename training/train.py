"""
人体运动姿态识别 — ResNet18 训练脚本
用法: python train.py md        → 单特征微多普勒
      python train.py rtdtat    → RT+DT+AT 三通道融合
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np, os, sys

# ============================================================
# 1. 配置 — 命令行切换数据集
# ============================================================
MODE = sys.argv[1] if len(sys.argv) > 1 else "md"

if MODE == "md":
    DATASET_PATH = r"D:\downlowd_cloud\方向2-雷达数据demo\training\dataset_single"
    INPUT_TYPE   = "gray"
    MODEL_NAME   = "resnet18_single_md.pth"
    print("===== 单特征：微多普勒 =====")
elif MODE == "rtdtat":
    DATASET_PATH = r"D:\downlowd_cloud\方向2-雷达数据demo\training\dataset_fused_rtdtat"
    INPUT_TYPE   = "rgb"
    MODEL_NAME   = "resnet18_fused_rtdtat.pth"
    print("===== RT+DT+AT 三通道融合 =====")
else:
    print(f"Unknown mode: {MODE}. Use 'md' or 'rtdtat'")
    sys.exit(1)

BATCH_SIZE   = 32
EPOCHS       = 40
LEARNING_RATE = 0.001
TRAIN_RATIO  = 0.75
IMG_SIZE     = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"设备: {DEVICE}")
print(f"数据集路径: {DATASET_PATH}")

# ============================================================
# 2. 数据加载和预处理
# ============================================================

# 根据输入类型选择不同的预处理
if INPUT_TYPE == "gray":
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
else:  # rgb — 三通道融合图，不需要 Grayscale
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

# ImageFolder 加载全部数据（先用统一 transform，后面覆盖）
dataset = ImageFolder(DATASET_PATH, transform=train_transform)
class_names = dataset.classes
num_classes = len(class_names)

print(f"\n类别 ({num_classes} 类): {class_names}")
print(f"总样本数: {len(dataset)}")
for i, name in enumerate(class_names):
    count = sum(1 for _, label in dataset.samples if label == i)
    print(f"  {name}: {count} 张")

# 训练/测试划分
n_train = int(len(dataset) * TRAIN_RATIO)
n_test  = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test],
                                    generator=torch.Generator().manual_seed(42))

# 测试集用 test_transform
test_set.dataset.transform = test_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\n训练集: {n_train} 张, 测试集: {n_test} 张")

# ============================================================
# 3. 构建 ResNet18 模型
# ============================================================

try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    print("  使用 ImageNet 预训练权重")
except Exception:
    model = models.resnet18(weights=None)
    print("  预训练下载失败，使用随机初始化")
model.fc = nn.Linear(512, num_classes)   # 改最后一层：512 → 你的类别数
model = model.to(DEVICE)

print(f"\n模型: ResNet18")
print(f"  最后一层: 512 → {num_classes}")

# ============================================================
# 4. 训练
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n开始训练 ({EPOCHS} epochs)...")
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 每个 epoch 评估一次
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    acc = correct / n_test
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  Loss: {avg_loss:.4f}  Test Acc: {acc:.2%}")

# ============================================================
# 5. 最终评估
# ============================================================

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        pred = outputs.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 准确率
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n========== 最终结果 ==========")
print(f"测试准确率: {accuracy:.2%}")

# 分类报告
print(f"\n各类别指标:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('预测类别')
ax.set_ylabel('真实类别')
ax.set_title(f'混淆矩阵 (准确率 {accuracy:.1%})')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(DATASET_PATH), f'confusion_matrix_{MODE}.png'),
            dpi=150, bbox_inches='tight')
print(f"\n混淆矩阵已保存")

# 保存模型
model_path = os.path.join(os.path.dirname(DATASET_PATH), MODEL_NAME)
torch.save(model.state_dict(), model_path)
print(f"模型已保存: {model_path}")
