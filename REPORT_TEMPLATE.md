# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Họ và tên: Nguyễn Minh Phượng
- Lớp: KHMT17-01
- Repo: `D:\HT\Nam3_KH3\Deep_Learning\csc4005-lab2-cnn-neu-1677030156NguyenMinhPhuongKHMT17-01-1`
- W&B project: `csc4005-lab2-neu-cnn` (run local, chưa sync online)

## 2. Bài toán
Phân loại ảnh grayscale lỗi bề mặt thép của bộ NEU-CLS thành 6 lớp: Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches.  
Dữ liệu chạy từ file: `D:\HT\Nam3_KH3\Deep_Learning\NEU-CLS.zip`.

## 3. Mô hình và cấu hình
### 3.1. MLP baseline từ Lab 1
Chưa có file kết quả Lab 1 trong repo hiện tại, nên không thể điền số liệu MLP chính xác.

### 3.2. CNN from scratch
- Run name: `cnn_small_baseline`
- Model: `cnn_small`
- Train mode: `scratch`
- Optimizer: `AdamW`
- LR: `0.001`, Weight decay: `0.0001`, Dropout: `0.3`
- Epochs: `20`, Batch size: `32`, Image size: `64`, Patience: `5`
- Augmentation: bật

### 3.3. Transfer learning
- Run name: `resnet18_transfer`
- Model: `resnet18`
- Train mode: `transfer` (freeze backbone, train classifier head)
- Optimizer: `AdamW`
- LR: `0.001`, Weight decay: `0.0001`, Dropout: `0.3`
- Epochs: `10`, Batch size: `32`, Image size: `128`, Patience: `3`
- Augmentation: bật

## 4. Bảng kết quả
| Model | Train mode | Best Val Acc | Test Acc | Epoch time (sec) | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | N/A | N/A | N/A | N/A | Chưa có số liệu trong repo Lab 2 |
| CNN-small | scratch | 0.9444 | 0.9481 | 7.47 | 32,614 | Học tốt, chạy nhanh/epoch, dao động val loss ở vài epoch |
| ResNet18 | transfer | 0.9667 | 0.9630 | 29.82 | 3,078 | Val/Test tốt hơn nhẹ, hội tụ nhanh theo số epoch nhưng tốn thời gian/epoch hơn |

## 5. Phân tích learning curves
- **CNN-small (scratch):** train loss giảm tốt, val acc đạt cao nhưng có vài điểm dao động mạnh (ví dụ val loss tăng đột ngột ở epoch 13 và 17), cho thấy có overfitting cục bộ.
- **ResNet18 (transfer):** val acc tăng nhanh lên >0.94 từ sớm, đường loss/acc ổn định hơn, khoảng cách train-val nhỏ hơn đa số epoch.
- Tổng thể: transfer learning cho đường học ổn định hơn theo epoch, còn scratch cho tốc độ mỗi epoch nhanh hơn đáng kể.

## 6. Confusion matrix và lỗi dự đoán sai
- **Best model theo validation:** `resnet18_transfer` (best val acc = 0.9667).
- Các nhầm lẫn chính của CNN scratch:
  - Inclusion bị nhầm sang Pitted_Surface.
  - Scratches bị nhầm sang Inclusion.
- Các nhầm lẫn chính của ResNet18 transfer:
  - Inclusion bị nhầm sang Pitted_Surface và Scratches.
  - Một số Crazing bị nhầm sang Patches.
- File kết quả đã lưu tại:
  - `outputs\cnn_small_baseline\`
  - `outputs\resnet18_transfer\`
  (bao gồm `best_model.pt`, `history.csv`, `curves.png`, `confusion_matrix.png`, `metrics.json`)

## 7. Kết luận
- **CNN có cải thiện so với MLP không?** Chưa kết luận định lượng được vì thiếu số liệu MLP Lab 1 trong repo hiện tại.
- **Transfer learning có tốt hơn không?** Có, theo metrics của Lab 2: ResNet18 transfer đạt best val acc và test acc cao hơn nhẹ so với CNN scratch.
- **Khi nào nên chọn transfer learning thay vì train from scratch?**
  - Nên chọn transfer khi cần chất lượng tốt nhanh trên dữ liệu không quá lớn.
  - Nên chọn scratch khi cần pipeline nhẹ, thời gian mỗi epoch nhanh, hoặc muốn mô hình đơn giản và chủ động hơn với miền dữ liệu đặc thù.
