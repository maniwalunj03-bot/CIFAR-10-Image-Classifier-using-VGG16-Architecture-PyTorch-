# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:07:22 2025

@author: Manisha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time, copy, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# VGG configutation dictionary
_VGG_CONFIGS = {
    "VGG11" : [1, 1, 2, 2, 2],
    "VGG13" : [2, 2, 2, 2, 2],
    "VGG16" : [2, 2, 3, 3, 3],
    "VGG19" : [2, 2, 4, 4, 4]
    }

# Channel progression per block
_BLOCK_CHANNELS = [64, 128, 256, 512, 512]

class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs, use_bn=False):
        """
       Build a block with `num_convs` convolution layers followed by one MaxPool.
       Each conv uses kernel_size=3, padding=1 (preserves spatial size).
       """
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                                    bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGG(nn.Module):
    def __init__(self, variant="VGG16", num_classes=10, use_bn=False, 
                 in_channels=3):
        """
        variant: "VGG11","VGG13","VGG16","VGG19"
        use_bn: whether to add BatchNorm after convs (commonly used: True)
        Designed for CIFAR-size inputs (32x32). We use AdaptiveAvgPool 
        before classifier.
        """
        super().__init__()
        assert variant in _VGG_CONFIGS, f"Unknown variant: {variant}"
        cfg = _VGG_CONFIGS[variant]
        
        layers = []
        prev_ch = in_channels
        for block_idx, num_convs in enumerate(cfg):
            out_ch = _BLOCK_CHANNELS[block_idx]
            layers.append(VGGBlock(prev_ch, out_ch, num_convs, use_bn))
            prev_ch = out_ch
            
        self.features = nn.Sequential(*layers)
        # Use global pooling to collapse HxW to 1x1 (works for variable 
        # input sizes, and reduces FC params)            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Small classifier for CIFAR; original VGG uses huge FCs for 
        # ImageNet (4096)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BLOCK_CHANNELS[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
            )
        self._init_weights()
        
    def forward(self, x):
        x = self.features(x)    # Conv block
        x = self.avgpool(x)     # global average pool (B, C, 1, 1)
        x = self.classifier(x)  # Classifier on channels
        
        return x
    
    def _init_weights(self):
        # Kaiming for convs, normal for linear biases = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            
# Training AND Evaluation
def get_CIFAR10_loaders(data_dir='./data', batch_size=128, val_split=0.1, 
                        augment=True):
    # Transforms tuned for CIFAR
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
            ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])
        
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    
    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf
        )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_tf
        )
    
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val], 
                                      generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()* images.size(0)
        _, preds = output.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    return running_loss/total, 100*correct/total

def evaluate(model, device, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss_sum += loss.item() * images.size(0)
            _, preds = output.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
    return loss_sum / total, 100 * correct/total

def train_loop(model, device, train_loader, val_loader, epochs=30, lr=0.01,
               weight_decay=1e-4, scheduler=None, patience=7,
               save_path="best_vgg.pth"):
    """
    Training loop with checkpointing and optional scheduler.
    Returns: best_model_wts, history dict
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                          weight_decay=weight_decay)
    if scheduler is None:
        # default conservative StepLR can be used, but user can pass 
        # ReduceLROnPlateau
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    best_wts = copy.deepcopy(model.state_dict())
    epochs_no_imp = 0
    history = {"train_loss":[], "train_accs":[], "val_loss":[], "val_accs":[]}
    
    start_time = time.time()
    for epoch in range(1, epochs+1):
        train_loss, train_accs = train_one_epoch(model, device, train_loader, 
                                                 optimizer, criterion)
        val_loss, val_accs = train_one_epoch(model, device, val_loader,
                                             optimizer, criterion)
         
        history["train_loss"].append(train_loss)
        history["train_accs"].append(train_accs)
        history["val_loss"].append(val_loss)
        history["val_accs"].append(val_accs)
        
        # scheduler step if using StepLR (call every epoch)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        improved = False
        if val_loss + 1e-5 < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, save_path)
            improved = True
            epochs_no_imp = 0
            print(f"[Epoch {epoch}] Improved val loss -> saved model. "
                  f"val_loss: {val_loss:.4f}")
        else:
            epochs_no_imp += 1
            print(f"[Epoch {epoch}] No improvement for {epochs_no_imp} "
                  f"epoch(s). val_loss: {val_loss:.4f}")
            
        print(f"Epoch {epoch}/{epochs}  Train: loss={train_loss:.4f}, "
               f"acc={train_accs:.2f}%  Val: loss={val_loss:.4f}, "
               f"acc={val_accs:.2f}%")
        
        torch.save(history, "vgg16_history.pt")
        
        if epochs_no_imp >= patience:
            print(f"Early stopping (patience={patience}) at epoch {epoch}")
            break
        
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f}s. Best val loss: "
          f"{best_val_loss:.4f}")
    model.load_state_dict(best_wts)
    history = torch.load("vgg16_history.pt")
    return model, history


# Example Usage

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_CIFAR10_loaders(batch_size=128,
                                                                augment=True)
    # Choose variant and whether to use BatchNorm
    model = VGG(variant="VGG16", num_classes=10, use_bn=True).to(device)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))
    model, history = train_loop(model, device, train_loader, val_loader,
                                epochs=60, lr=0.01, weight_decay=5e-4,
                                save_path="best_vgg16_cifar.pth",
                                scheduler=optim.lr_scheduler.ReduceLROnPlateau
                                (optim.SGD(model.parameters(), lr=0.01, 
                                           momentum=0.9, weight_decay=5e-4), 
                                 mode='min', factor=0.5, patience=3),
                                patience=8)
    test_loss, test_acc = evaluate(model, device, test_loader, 
                                   nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
# Visualization
history = torch.load("vgg16_history.pt")

# 1. Loss curves
plt.figure(figsize=(8,5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("VGG16 Training vs Validation Loss")
plt.legend()
plt.show()

# 2. Accuracy curves
plt.figure(figsize=(8,5))
plt.plot(history["train_accs"], label="Train Acc")
plt.plot(history["val_accs"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("VGG16 Training vs Validation Accuracy")
plt.legend()
plt.show()

# Confusion Matrix
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print(classification_report(all_labels, all_preds, digits=4))

# Heatmap 
# CIFAR-10 class names
classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16 on CIFAR10")
plt.show()


#Grad-CAM Visualization for One Image

# Pick one batch
images, labels = next(iter(test_loader))
img = images[0:1].to(device)

model.eval()

# ---- Forward pass to get feature maps ----
gradients = []
activations = []

def save_gradient(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.append(output)

# Register the last conv layer
target_layer = model.features[-1]
target_layer.register_forward_hook(save_activation)
target_layer.register_full_backward_hook(save_gradient)

# Forward pass
output = model(img)
pred_class = output.argmax(dim=1).item()

# Backprop for Grad-CAM
model.zero_grad()
output[0, pred_class].backward()

# Extract activations & gradients
acts = activations[0].cpu().detach().numpy()[0]
grads = gradients[0].cpu().detach().numpy()[0]

weights = grads.mean(axis=(1,2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * acts[i]

# Normalize CAM
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (32,32))
cam = cam / cam.max()

# Original image
img_np = images[0].permute(1,2,0).numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap = heatmap[..., ::-1] / 255.0

# Overlay
overlay = 0.5 * img_np + 0.5 * heatmap

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.show()
