"""
Phân tích mẫu dự đoán sai của best model.
Dùng: python -m src.analyze_errors --run_name <tên_run> --data_dir <đường_dẫn>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import create_dataloaders
from src.model import build_model
from src.train import resolve_input_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path('outputs') / args.run_name
    metrics_path = output_dir / 'metrics.json'
    model_path = output_dir / 'best_model.pt'

    with metrics_path.open(encoding='utf-8') as f:
        metrics = json.load(f)

    model_name = metrics['model_name']
    train_mode = metrics['train_mode']
    num_channels = metrics['num_channels']
    normalization = metrics['normalization']
    class_names = metrics['class_names']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = create_dataloaders(
        data_dir=args.data_dir,
        img_size=128 if train_mode != 'scratch' else 64,
        batch_size=32,
        val_size=0.15,
        test_size=0.15,
        random_state=args.seed,
        augment=False,
        num_workers=0,
        num_channels=num_channels,
        normalization=normalization,
    )

    model = build_model(
        model_name=model_name,
        train_mode=train_mode,
        num_classes=len(class_names),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wrong_images, wrong_preds, wrong_trues = [], [], []
    with torch.no_grad():
        for x, y in data.test_loader:
            logits = model(x.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            mask = preds != y
            for img, pred, true in zip(x[mask], preds[mask], y[mask]):
                wrong_images.append(img)
                wrong_preds.append(pred.item())
                wrong_trues.append(true.item())
            if len(wrong_images) >= args.n_samples:
                break

    n = min(args.n_samples, len(wrong_images))
    if n == 0:
        print('Không có mẫu nào dự đoán sai trên test set!')
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for ax, img, pred, true in zip(axes, wrong_images[:n], wrong_preds[:n], wrong_trues[:n]):
        # img shape: C x H x W
        if img.shape[0] == 1:
            show = img[0].numpy()
            ax.imshow(show, cmap='gray')
        else:
            # denormalize ImageNet
            show = (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
            ax.imshow(show)
        ax.set_title(f'True: {class_names[true]}\nPred: {class_names[pred]}', fontsize=9)
        ax.axis('off')

    plt.suptitle(f'Mẫu dự đoán sai – {args.run_name}', fontsize=11)
    plt.tight_layout()
    save_path = output_dir / 'wrong_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Đã lưu {n} mẫu sai vào: {save_path}')


if __name__ == '__main__':
    main()
