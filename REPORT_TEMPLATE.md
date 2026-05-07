# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Họ và tên: Nguyễn Minh Phương
- Lớp: KHMT17-01
- MSSV: 1677030156
- Repo: csc4005-lab2-cnn-neu-1677030156NguyenMinhPhuongKHMT17-01-1
- W&B project: https://wandb.ai/shootingsaviorstar-discord/csc4005-lab2-neu-cnn

## 2. Bài toán
Phân loại 6 loại lỗi bề mặt thép (Crazing, Inclusion, Patches, Pitted\_Surface, Rolled-in\_Scale, Scratches) từ ảnh grayscale thuộc bộ dữ liệu **NEU Surface Defect Database (NEU-CLS)**. Tổng cộng 1.800 ảnh, chia theo tỉ lệ 70/15/15 (train/val/test), mỗi lớp 300 ảnh.

## 3. Mô hình và cấu hình

### 3.1. MLP baseline từ Lab 1
- Kiến trúc MLP 3 lớp ẩn (flatten ảnh → vector 1D → FC layers).
- Input: ảnh grayscale flatten về vector 1D.
- Hạn chế: mất hoàn toàn cấu trúc không gian; số tham số lớn.

### 3.2. CNN from scratch (`cnn_small`)
- 3 × ConvBlock (Conv2d → BatchNorm → ReLU → MaxPool): 16 → 32 → 64 channels.
- AdaptiveAvgPool2d → FC(128) → Dropout(0.3) → FC(6).
- **Cấu hình:** `img_size=64`, `lr=0.001`, `optimizer=adamw`, `weight_decay=1e-4`, `dropout=0.3`, `epochs=20`, `batch_size=32`, `augment=True`, `scheduler=plateau`, `patience=5`.
- Tổng tham số: **32.614** (toàn bộ có thể train).

### 3.3. Transfer learning – ResNet18 (frozen backbone)
- ResNet18 pretrained ImageNet, **freeze** toàn bộ backbone, thay FC cuối → Dropout(0.3) + Linear(512, 6).
- Input chuẩn hoá ImageNet (3 channel, `img_size=128`).
- **Cấu hình:** `lr=0.001`, `optimizer=adamw`, `weight_decay=1e-4`, `dropout=0.3`, `epochs=10`, `batch_size=32`, `patience=3`, `augment=True`.
- Tham số trainable: **3.078** (chỉ classifier head).

### 3.4. Fine-tune – ResNet18 (toàn bộ backbone mở)
- ResNet18 pretrained ImageNet, **mở toàn bộ** các lớp để train.
- **Cấu hình:** `lr=0.0001`, `optimizer=adamw`, `weight_decay=1e-4`, `dropout=0.3`, `epochs=10`, `batch_size=16`, `patience=3`, `augment=True`.
- Tham số trainable: **11.179.590**.

## 4. Bảng kết quả

| Model | Train mode | Best Val Acc | Test Acc | Avg Epoch Time | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP (Lab 1) | scratch | ~85% | ~84% | ~1 s | ~200k | Flatten mất cấu trúc không gian |
| CNN-small | scratch | **94.44%** | **94.81%** | 4.20 s | 32.614 | Nhẹ, hội tụ tốt sau 20 epoch |
| ResNet18 | transfer (frozen) | 96.67% | 96.30% | 12.58 s | 3.078 | Nhanh hội tụ, ít tham số train |
| ResNet18 | finetune (full) | **100%** | **100%** | 34.99 s | 11.179.590 | Tốt nhất, nhưng nặng hơn |

> W&B Dashboard: https://wandb.ai/shootingsaviorstar-discord/csc4005-lab2-neu-cnn

## 5. Phân tích learning curves

### CNN from scratch (`cnn_small_baseline`)
- `train_loss` giảm đều từ 1.40 → 0.17 qua 20 epoch.
- `val_loss` dao động nhưng xu hướng giảm (0.13 ở epoch 16), mô hình không bị overfitting nghiêm trọng nhờ Dropout + augmentation.
- `val_acc` đạt đỉnh **94.4%** ở epoch 16 và ổn định.
- Scheduler `ReduceLROnPlateau` giảm lr từ 0.001 → 0.0005 → 0.00025 giúp mô hình hội tụ tinh.

### ResNet18 transfer (frozen)
- `val_acc` tăng mạnh ngay từ epoch 1 (~79.6%) vì backbone đã có sẵn đặc trưng ảnh.
- Hội tụ nhanh hơn CNN scratch (10 epoch đủ), `val_acc` đạt **96.7%** epoch cuối.
- Không có hiện tượng overfitting rõ ràng.

### ResNet18 finetune (full)
- `val_acc` đạt **100%** từ epoch 2, `val_loss` giảm rất nhanh về ~0.002.
- Early stopping kích hoạt ở epoch 8, không cần chạy thêm.
- Learning curve rất dốc và ổn định – dấu hiệu fine-tune hiệu quả khi dữ liệu đủ và backbone đã pretrained.

## 6. Confusion matrix và lỗi dự đoán sai

### CNN-small (test set, 270 mẫu)
- Phân loại đúng hoàn toàn: Crazing, Patches, Rolled-in\_Scale.
- Lỗi chủ yếu: **Inclusion** bị nhầm sang Pitted\_Surface (5/45), **Pitted\_Surface** bị nhầm sang Inclusion (2/45) và Scratches (1/45), **Scratches** bị nhầm sang Inclusion (4/45).
- Các cặp dễ nhầm: Inclusion ↔ Pitted\_Surface (texture tương tự), Scratches ↔ Inclusion.

### ResNet18 transfer (test set, 270 mẫu)
- Lỗi ít hơn: Inclusion bị nhầm (7 mẫu), Crazing bị nhầm sang Patches (2 mẫu).
- Rolled-in\_Scale phân loại hoàn hảo.

### ResNet18 finetune (test set, 270 mẫu)
- **0 lỗi** – confusion matrix hoàn toàn là đường chéo.

## 7. Kết luận

### CNN có cải thiện so với MLP không?
Có. CNN-small đạt **94.8% test accuracy** so với ~84% của MLP baseline. Lý do: CNN giữ lại cấu trúc không gian của ảnh thông qua convolution + pooling, weight sharing giúp số tham số ít hơn nhiều, trong khi MLP chỉ nhìn thấy dãy số phẳng.

### Transfer learning có tốt hơn không?
Có. ResNet18 (frozen) đạt **96.3%** chỉ với 10 epoch và 3.078 tham số trainable. ResNet18 (finetune) đạt **100%** chỉ trong 8 epoch. Cả hai đều vượt CNN scratch.

### Khi nào nên chọn transfer learning thay vì train from scratch?
- **Nên dùng transfer learning khi:**
  - Dữ liệu không quá lớn (NEU-CLS chỉ 1.800 ảnh).
  - Cần kết quả tốt nhanh với tài nguyên tính toán hạn chế.
  - Domain ảnh không quá khác so với ImageNet (ảnh grayscale bề mặt thép vẫn có texture học được).
  - Chỉ muốn train classifier head → ít tham số, ít nguy cơ overfitting.
- **Khi nào chưa chắc cần dùng:**
  - Dữ liệu rất lớn và domain khác xa ImageNet (vd: ảnh y tế chuyên biệt, ảnh vệ tinh).
  - Backbone không phù hợp với kích thước/loại input.
  - Tài nguyên đủ để train from scratch và muốn kiến trúc nhỏ gọn hơn.

**Kết luận tổng quát:** Với bài toán NEU-CLS, **ResNet18 finetune** là best model (test acc = 100%), nhưng nếu ưu tiên tốc độ train và model nhẹ, **CNN-small scratch** (94.8%, 32k params, 4.2s/epoch) là lựa chọn thực tế hơn.
