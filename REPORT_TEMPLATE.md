# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Họ và tên: Nguyễn Minh Phượng
- Lớp: KHMT17-01
- Repo: `D:\HT\Nam3_KH3\Deep_Learning\csc4005-lab2-cnn-neu-1677030156NguyenMinhPhuongKHMT17-01-1`
- W&B project: `csc4005-lab2-neu-cnn`

## 2. Bài toán
Phân loại ảnh grayscale lỗi bề mặt thép của bộ NEU-CLS thành 6 lớp: Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches.  
Dữ liệu chạy từ file: `D:\HT\Nam3_KH3\Deep_Learning\NEU-CLS.zip` (3610 files, 6 classes × ~300 ảnh/class).

## 3. Mô hình và cấu hình

### 3.1. MLP baseline từ Lab 1
Lấy từ repo Lab 1: `csc4005-lab1-neu-mlp-1677030156NguyenMinhPhuongKHMT17-01`  
Best config của Lab 1: **run_b_sgd** (SGD, LR=0.01, Dropout=0.3, Weight Decay=0.0)
- **Model:** MLP (flatten 64×64 → hidden layers → 6 classes)
- **Optimizer:** SGD | **LR:** 0.01 | **Weight decay:** 0.0 | **Dropout:** 0.3
- **Epochs:** 20 | **Batch size:** 32 | **Image size:** 64 | **Patience:** 5
- **Best val acc:** 48.89% | **Test acc:** 47.41%
- **Nhận xét:** MLP flatten ảnh thành vector, mất hoàn toàn cấu trúc không gian → accuracy thấp (~47%)

### 3.2. CNN from scratch (`cnn_small`)
- **Run name:** `cnn_small_baseline`
- **Model:** `SmallCNN` gồm 3 ConvBlock (Conv2d → BatchNorm → ReLU → MaxPool) + AdaptiveAvgPool + Classifier (Linear 64→128 → ReLU → Dropout → Linear 128→6)
- **Train mode:** `scratch`
- **Optimizer:** AdamW | **LR:** 0.001 | **Weight decay:** 0.0001 | **Dropout:** 0.3
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2)
- **Epochs:** 20 | **Batch size:** 32 | **Image size:** 64 | **Patience:** 5
- **Input:** 1 channel (grayscale) | **Normalization:** none
- **Augmentation:** bật (rotation ±8°, translate ±5%, brightness/contrast jitter ±10%)

### 3.3. Transfer Learning (`resnet18`)
- **Run name:** `resnet18_transfer`
- **Model:** ResNet18 (pretrained ImageNet) + custom classifier head (Dropout → Linear 512→6)
- **Train mode:** `transfer` (freeze toàn bộ backbone, chỉ train classifier head)
- **Optimizer:** AdamW | **LR:** 0.001 | **Weight decay:** 0.0001 | **Dropout:** 0.3
- **Scheduler:** ReduceLROnPlateau
- **Epochs:** 10 | **Batch size:** 32 | **Image size:** 128 | **Patience:** 3
- **Input:** 3 channels (grayscale → duplicate 3 kênh) | **Normalization:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation:** bật

## 4. Bảng kết quả

| Model | Train mode | Best Val Acc | Best Val Loss | Test Acc | Test Loss | Avg Epoch Time (sec) | Total Params | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLP (Lab 1 best) | scratch | 0.4889 | 1.4432 | 0.4741 | 1.4055 | N/A | ~2.2M | ~2.2M | Flatten ảnh → mất spatial info, acc rất thấp |
| CNN-small | scratch | **0.9444** | 0.1298 | **0.9481** | 0.1375 | **4.26** | 32,614 | 32,614 | Toàn bộ params đều trainable, chạy nhanh/epoch |
| ResNet18 | transfer | **0.9667** | 0.1506 | **0.9630** | 0.1567 | **11.71** | 11,179,590 | 3,078 | Chỉ train 3,078 params (0.03%), nhưng đạt acc cao hơn |

### Phân tích chi tiết per-class (Test set, mỗi class 45 mẫu):

| Class | CNN-small Precision | CNN-small Recall | ResNet18 Precision | ResNet18 Recall |
|---|---:|---:|---:|---:|
| Crazing | 1.000 | 1.000 | 1.000 | 0.956 |
| Inclusion | 0.870 | 0.889 | 0.974 | 0.844 |
| Patches | 1.000 | 1.000 | 0.957 | 1.000 |
| Pitted_Surface | 0.889 | 0.889 | 0.900 | 1.000 |
| Rolled-in_Scale | 0.957 | 1.000 | 1.000 | 1.000 |
| Scratches | 0.976 | 0.911 | 0.957 | 0.978 |

## 5. Phân tích learning curves

### 5.1. CNN-small (scratch) — 20 epochs
- **Train loss** giảm đều từ 1.40 → 0.17 qua 20 epoch.
- **Val loss** có xu hướng giảm nhưng **dao động mạnh**: tăng đột biến tại epoch 13 (val_loss = 1.15, val_acc rơi về 0.685). Đây là dấu hiệu **overfitting cục bộ** — mô hình nhạy cảm với từng batch validation.
- **Val acc** đạt đỉnh 0.9444 tại epoch 16, sau đó ổn định ~0.937.
- **Learning rate** giảm từ 0.001 → 0.0005 (epoch 8) → 0.00025 (epoch 19) nhờ scheduler.
- Khoảng cách train-val loss co hẹp dần ở cuối nhưng val vẫn dao động.

### 5.2. ResNet18 (transfer) — 10 epochs
- **Train loss** giảm mượt từ 1.35 → 0.23.
- **Val loss** giảm rất ổn định từ 0.94 → 0.15, **không có spike lớn**.
- **Val acc** tăng nhanh: đạt >0.91 từ epoch 2, đạt đỉnh 0.9667 tại epoch 10.
- **Đường học ổn định hơn nhiều** so với scratch — backbone pretrained đã cung cấp features tốt sẵn.
- **Val acc luôn cao hơn train acc** ở hầu hết epoch → backbone frozen tạo ra features tốt mà classifier head dễ tối ưu.

### 5.3. So sánh tổng thể
- Transfer learning cho **đường học mượt, ổn định** hơn đáng kể.
- Scratch cho **tốc độ mỗi epoch nhanh hơn ~2.75×** (4.26s vs 11.71s) do model nhỏ hơn nhiều.
- Scratch cần nhiều epoch hơn để hội tụ (20 vs 10), nhưng tổng thời gian train gần nhau (85s vs 117s).

## 6. Confusion matrix và lỗi dự đoán sai

### 6.1. Best model: `resnet18_transfer` (best val acc = 0.9667)

### 6.2. Các nhầm lẫn chính của CNN-small (scratch):
- **Inclusion → Pitted_Surface:** 5 mẫu bị nhầm (chiếm 11.1% Inclusion)
- **Pitted_Surface → Inclusion:** 2 mẫu, → Rolled-in_Scale: 2 mẫu, → Scratches: 1 mẫu
- **Scratches → Inclusion:** 4 mẫu bị nhầm
- Crazing, Patches, Rolled-in_Scale: phân loại hoàn hảo 100%

### 6.3. Các nhầm lẫn chính của ResNet18 (transfer):
- **Inclusion → Pitted_Surface:** 5 mẫu, → Scratches: 2 mẫu (Inclusion vẫn là class khó nhất)
- **Crazing → Patches:** 2 mẫu bị nhầm
- **Scratches → Inclusion:** 1 mẫu
- Patches, Pitted_Surface, Rolled-in_Scale: phân loại hoàn hảo 100%

### 6.4. Nhận xét
- **Inclusion** là lớp khó nhất cho cả 2 mô hình — dễ nhầm với Pitted_Surface và Scratches do đặc trưng bề mặt tương tự.
- ResNet18 cải thiện rõ ở Pitted_Surface (100% recall vs 88.9%) và Scratches (97.8% vs 91.1%).
- CNN-small lại tốt hơn ở Crazing (100% vs 95.6%).

## 7. Output files
Kết quả đã lưu tại:
- `outputs/cnn_small_baseline/` — best_model.pt, history.csv, curves.png, confusion_matrix.png, metrics.json
- `outputs/resnet18_transfer/` — best_model.pt, history.csv, curves.png, confusion_matrix.png, metrics.json

## 8. Kết luận

### 8.1. CNN có cải thiện so với MLP không?
**Có, cải thiện cực kỳ rõ rệt.** Dựa trên số liệu thực nghiệm từ Lab 1 và Lab 2:
- MLP best (SGD): test acc = **47.41%** → CNN-small: test acc = **94.81%** → cải thiện **+47.4 điểm phần trăm** (~gấp đôi)
- MLP best val loss = 1.4432 → CNN-small best val loss = 0.1298 → giảm **~11 lần**
- CNN chỉ cần **32,614 params** trong khi MLP cần ~2.2M params → nhẹ hơn ~67 lần

**Lý do:** CNN giữ cấu trúc không gian (spatial structure) của ảnh, kernel trượt học đặc trưng cục bộ (local features), weight sharing giảm số tham số. MLP phải flatten ảnh thành vector 1D → mất hoàn toàn thông tin vị trí pixel, dẫn đến không thể học được các pattern hình học (cạnh, texture, vết nứt) vốn là yếu tố quyết định trong phân loại lỗi bề mặt thép.

### 8.2. Transfer learning có tốt hơn không?
**Có**, theo số liệu thực nghiệm:
- ResNet18 transfer đạt **test acc 96.3%** vs CNN-small **94.8%** (+1.5%)
- Val acc cao hơn: **96.67%** vs **94.44%** (+2.23%)
- Đường học ổn định hơn, không có spike
- Chỉ cần train **3,078 params** (0.03% tổng model) thay vì 32,614 params

### 8.3. Khi nào nên chọn transfer learning?
**Nên dùng transfer learning khi:**
- Dữ liệu không quá lớn (NEU-CLS chỉ có ~1800 ảnh) — backbone pretrained trên ImageNet đã học sẵn rất nhiều features hữu ích
- Cần chất lượng tốt nhanh với ít epoch
- Muốn learning curves ổn định, dễ debug
- Domain dữ liệu gần với ImageNet (ảnh tự nhiên, texture)

**Nên dùng scratch khi:**
- Cần pipeline nhẹ, thời gian mỗi epoch nhanh (4.26s vs 11.71s)
- Domain dữ liệu rất khác ImageNet (ảnh y tế, ảnh vệ tinh đặc thù)
- Muốn kiểm soát hoàn toàn kiến trúc mô hình
- Không có torchvision/pretrained weights sẵn

### 8.4. Tổng kết

| So sánh | MLP (Lab 1) | CNN scratch (Lab 2) | ResNet18 transfer (Lab 2) |
|---|---:|---:|---:|
| Test Accuracy | 47.41% | 94.81% | **96.30%** |
| Improvement vs MLP | — | +47.4% | **+48.9%** |
| Trainable Params | ~2.2M | 32,614 | 3,078 |
| Learning Stability | Dao động mạnh | Dao động vừa | Rất ổn định |

- **MLP → CNN:** Bước nhảy vọt lớn nhất (+47%), chứng minh CNN phù hợp hơn MLP cho bài toán ảnh.
- **CNN scratch → Transfer:** Cải thiện thêm +1.5% accuracy, đường học ổn định hơn, chỉ cần train rất ít params.
- Transfer learning thắng trên cả **accuracy** và **stability**. CNN-small scratch vẫn đạt kết quả tốt (94.8%) với model cực nhẹ (32K params) và train nhanh. Lựa chọn phụ thuộc vào ưu tiên: **accuracy** → transfer, **efficiency** → scratch.
