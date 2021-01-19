import os
import sys
import cv2
import time
import mmcv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from crfseg import CRF

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.datasets import CityscapesDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.datasets import build_dataloader
from mmseg.core.evaluation import mean_iou

repo_path = Path(".").absolute().parent

if os.system == "nt":
    data_path = Path("D:\Datas\parking_violation")
else:
    data_path = repo_path.parent / "data" / "parking_violation"
sys.path.append(str(repo_path))


model_config = "deeplabv3plus"
backbone_config = "r50-d8"
backbone_dict = {
    "r50-d8": {
        "config_dir": "deeplabv3plus",
        "config": "deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py",
        "checkpoint": "deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth"
    },
    "r18-d8": {
        "config_dir": "deeplabv3plus",
        "config": "deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py",
        "checkpoint": "deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth"
    },
    "resnest": {
        "config_dir": "resnest",
        "config": "deeplabv3plus_s101-d8_512x1024_80k_cityscapes.py",
        "checkpoint": "deeplabv3plus_s101-d8_512x1024_80k_cityscapes_20200807_144429-1239eb43.pth"
    } 
}


config_path = repo_path.parent / "mmsegmentation" / "configs" / backbone_dict[backbone_config]["config_dir"]
checkpoint_path = repo_path.parent / "data" / "mmseg" / "checkpoints"
if not checkpoint_path.exists():
    checkpoint_path.mkdir(parents=True)

config_file = str(config_path / backbone_dict[backbone_config]["config"])
checkpoint_file = str(checkpoint_path / backbone_dict[backbone_config]["checkpoint"])
crf_checkpoint_file = f"crf_{backbone_config}.pth"

# Preprocess video: Origin size is 720x1280
height, width = 480, 640
video_name = "sample1"
# height, width = 720, 1280
video_path = str(data_path / "origin" / f"{video_name}.mp4")
resized_video_path = str(data_path / f"{video_name}_{height}x{width}.mp4")
resized_frames_path = data_path / f"{video_name}_{height}x{width}"
if not resized_frames_path.exists():
    resized_frames_path.mkdir()

frames_path = resized_frames_path / "img_dir"

cityspaces_path = repo_path.parent / "data" / "cityscapes"
device = "cuda"
batch_size = 2

seg_model = init_segmentor(config_file, checkpoint_file, device=device)
crf = CRF(n_spatial_dims=2, returns="log-proba").to(device)

cfg = seg_model.cfg
train_dataset = CityscapesDataset(
    data_root=cityspaces_path, 
    pipeline=cfg.data.train.pipeline, 
    img_dir=cfg.data.train.img_dir, 
    ann_dir=cfg.data.train.ann_dir, 
    test_mode=False
)

val_dataset = CityscapesDataset(
    data_root=cityspaces_path, 
    pipeline=cfg.data.val.pipeline, 
    img_dir=cfg.data.val.img_dir, 
    ann_dir=cfg.data.val.ann_dir, 
    test_mode=False
)

train_loader = build_dataloader(
    train_dataset, samples_per_gpu=batch_size, workers_per_gpu=0, dataloader_type="DataLoader")

val_loader = build_dataloader(
    val_dataset, samples_per_gpu=batch_size, workers_per_gpu=0, dataloader_type="DataLoader")

# Freeze Seg Network
for param in seg_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(crf.parameters(), lr=1e-3)

def get_evaluations(log_proba, seg, n_classes, ignore_idx):
    res = log_proba.argmax(1)
    miou, cate_acc, cate_iou = mean_iou(
        results=res,
        gt_seg_maps=seg, 
        num_classes=n_classes, 
        ignore_index=ignore_idx, 
        nan_to_num=0.0
    )
    # record
    return miou, cate_acc, cate_iou

def train_crf(train_loader, seg_model, crf, optimizer):
    batch_size = train_loader.batch_size
    train_dst = train_loader.dataset
    total_loss = 0
    total_miou = 0
    total_cate_acc = np.zeros(len(train_dst.CLASSES))
    total_cate_iou = np.zeros(len(train_dst.CLASSES))
    n_classes = len(train_dst.CLASSES)
    ignore_idx = train_dst.ignore_index
    n = 0
    
    pbar = tqdm(desc="[Train]", total=len(train_loader)*batch_size)
    seg_model.eval()
    crf.train()
    
    for data in train_loader:
        img = data["img"].data[0].to(device)
        img_meta = data["img_metas"].data[0]
        seg = data["gt_semantic_seg"].data[0].squeeze(1).to(device)

        optimizer.zero_grad()
        x = seg_model.inference(img, img_meta, rescale=False)
        log_proba = crf(x, display_tqdm=False)  # (B, K, H, W)
        loss = F.nll_loss(log_proba, seg, ignore_index=ignore_idx, reduction="mean")
        loss.backward()
        optimizer.step()

        # evaluations
        miou, cate_acc, cate_iou = get_evaluations(
            log_proba.cpu(), seg.cpu(), n_classes, ignore_idx
        )

        # record
        total_loss += loss.item()
        total_miou += miou
        total_cate_acc += cate_acc.round(4)
        total_cate_iou += cate_iou.round(4)
        
        n += 1

        pbar.update(batch_size)
        pbar.set_description(f"[Train] Loss {loss.item():.4f} | Mean IoU {miou:.4f}")
    pbar.close()
    
    return crf, total_loss/n, total_miou/n, total_cate_acc/n, total_cate_iou/n

def val_crf(val_loader, seg_model, crf):
    batch_size = val_loader.batch_size
    val_dst = val_loader.dataset
    total_loss = 0
    total_miou = 0
    total_cate_acc = np.zeros(len(val_dst.CLASSES))
    total_cate_iou = np.zeros(len(val_dst.CLASSES))
    n_classes = len(val_dst.CLASSES)
    ignore_idx = val_dst.ignore_index
    n = 0
    
    pbar = tqdm(desc="[Valid]", total=len(val_loader)*batch_size)
    seg_model.eval()
    crf.eval()
    with torch.no_grad():
        for data in val_loader:
            img = data["img"][0].data.to(device)
            img_meta = data["img_metas"][0].data[0]
            seg = data["gt_semantic_seg"][0].data.squeeze(1).long().to(device)

            x = seg_model.inference(img, img_meta, rescale=False)
            log_proba = crf(x, display_tqdm=False)
            loss = F.nll_loss(log_proba, seg, ignore_index=ignore_idx, reduction="mean")
            # evaluations
            miou, cate_acc, cate_iou = get_evaluations(
                log_proba.cpu(), seg.cpu(), n_classes, ignore_idx
            )
            # record
            total_loss += loss.item()
            total_miou += miou
            total_cate_acc += cate_acc.round(4)
            total_cate_iou += cate_iou.round(4)

            n += 1

            pbar.update(batch_size)
            pbar.set_description(f"[Valid] Loss {loss.item():.4f} | Mean IoU {miou:.4f}")
    pbar.close()
    
    return crf, total_loss/n, total_miou/n, total_cate_acc/n, total_cate_iou/n

n_step = 10
best_miou = 0.0
best_loss = 999999
crf_checkpoint_file = f"crf_{backbone_config}.pth"
sv_eval_path = Path("./eval_result.txt")
eval_file = sv_eval_path.open("w", encoding="utf-8")
eval_file.write("\t".join(train_dataset.CLASSES) + "\n")

for step in range(n_step):
    crf, train_loss, train_miou, train_cate_acc, train_cate_iou = train_crf(
        train_loader, seg_model, crf, optimizer)
    torch.cuda.empty_cache()
    crf, val_loss, val_miou, val_cate_acc, val_cate_iou = val_crf(
        val_loader, seg_model, crf)
    print(f"[{step+1}/{n_step}] Train Loss: {train_loss:.4f} | Train Mean IoU: {train_miou:.4f}")
    print(f"[{step+1}/{n_step}] Val Loss: {val_loss:.4f} | Val Mean IoU: {val_miou:.4f}")
    # Save 
    if val_loss < best_loss:
        best_loss = val_loss
        best_miou = val_miou
        best_cate_acc = val_cate_acc
        best_cate_iou = val_cate_iou
        torch.save(crf.state_dict(), crf_checkpoint_file)
        print("[INFO] Saved")
        
eval_file.write("\t".join([f"{acc:.4f}" for acc in best_cate_acc]) + "\n")
eval_file.write("\t".join([f"{iou:.4f}" for iou in best_cate_iou]) + "\n")
eval_file.close()