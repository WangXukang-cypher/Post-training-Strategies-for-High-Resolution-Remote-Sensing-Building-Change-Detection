# 运行的方式：python retrain.py --data_root /fast/Wang/Chaofen/Smashiki_256QuaterCD --save_path /fast/Wang/Chaofen/S128exp/WHU-CD_iter_200000_lr_0.0002/retrained_best.pth
#   h
# retrain.py
# BCD 后训练（不重训）：在 best_model 基础上，以 KL 保护 + DPO 偏好 + 边界/误检约束 做小步微调
# 仅需指定 --data_root 和 --save_path；其它参数已在文件中固定
# 目录结构要求：
# data_root/
#   ├── train/{t1,t2,label}
#   ├── val/{t1,t2,label}
#   └── test/{t1,t2,label}

import os, glob, argparse, math
from typing import List
import numpy as np
import cv2
from PIL import Image
import time


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 官方评估用到的模块
import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter

# 模型
from model.trainer import Trainer

# -------- 固定参数（根据需要改） --------
DEFAULT_CKPT  = "/fast/Wang/Change3D/S128exp/WHU-CD_iter_200000_lr_0.0002/best_model.pth"
BATCH_SIZE    = 8
EPOCHS        = 32
# epoch = 8,16,32
LR            = 2e-5
WARMUP_STEPS  = 500
MIN_LR_FACTOR = 0.2
EMA_DECAY     = 0.999
WORKERS       = 4

# 损失权重
LAMBDA_SUP = 1.0
LAMBDA_BND = 0.5     # 边更直 → 可稍大
LAMBDA_FP  = 0.5     # 抑制误检 → 可稍大
LAMBDA_DPO = 1
BETA_DPO   = 0.1
BETA_KL    = 3e-3    # 更“保守”的微调半径

FOCAL_ALPHA  = 0.25
FOCAL_GAMMA  = 2.0
FOCAL_WEIGHT = 0.25

# 只训练 decoder（如需更激进，把 BACKBONE_LAST_STAGE_TRAIN=True）
BACKBONE_LAST_STAGE_TRAIN = False

# 子目录名
PRE_DIR   = "t1"
POST_DIR  = "t2"
MASK_DIR  = "label"

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

def _list_images(d: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(d, f'*{ext}'))
    return sorted(files)

# ====== 官方评估用的 DataLoader（RSDataset + 官方 val_transform）======
def build_eval_loader(args, split):
    _, val_tf = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    ds = RSDataset.BCDDataset(file_root=args.data_root, split=split, transform=val_tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=WORKERS, pin_memory=True)

def evaluate_official(policy, loader):
    policy.eval()
    meter = ConfuseMatrixMeter(n_class=2)   # 二类：0/1
    with torch.no_grad():
        for img, target in loader:
            pre  = img[:, 0:3].cuda().float()
            post = img[:, 3:6].cuda().float()
            gt   = target.cuda().float()
            out  = policy.update_bcd(pre, post)   # 输出可为logits或概率
            pred = (out > 0.5).long().cpu().numpy()  # 与官方脚本一致：0.5阈值
            meter.update_cm(pr=pred, gt=gt.cpu().numpy())
    scores = meter.get_scores()
    policy.train()
    return scores['IoU'], scores['F1'], scores['precision'], scores['recall']

def sweep_thresholds_official(policy, loader, name="val"):
    """不改变口径，只打印在不同阈值下的F1，用于研究/比对"""
    policy.eval()
    ths = np.linspace(0.05, 0.95, 19)
    best = (-1, 0.5, 0, 0, 0)  # (F1, th, IoU, P, R)
    with torch.no_grad():
        for th in ths:
            meter = ConfuseMatrixMeter(n_class=2)
            for img, target in loader:
                pre  = img[:, 0:3].cuda().float()
                post = img[:, 3:6].cuda().float()
                gt   = target.cuda().float()
                out  = policy.update_bcd(pre, post)
                pred = (out > th).long().cpu().numpy()
                meter.update_cm(pr=pred, gt=gt.cpu().numpy())
            sc = meter.get_scores()
            if sc['F1'] > best[0]:
                best = (sc['F1'], th, sc['IoU'], sc['precision'], sc['recall'])
    print(f"[SWEEP/{name}] bestF1={best[0]:.4f} @th={best[1]:.2f} | IoU={best[2]:.4f} P={best[3]:.4f} R={best[4]:.4f}")
    policy.train()
    return best

def evaluate_with_ema(policy, ema, loader):
    ema.apply_shadow(policy)
    scores = evaluate_official(policy, loader)
    ema.restore(policy)
    return scores

# ====== 训练用（小而精）Dataset：保持你之前的实现 ======
class BCDPairDataset(Dataset):
    def __init__(self, args, root, split='train'):
        self.args = args
        self.split = split
        self.pre_dir  = os.path.join(root, split, PRE_DIR)
        self.post_dir = os.path.join(root, split, POST_DIR)
        self.mask_dir = os.path.join(root, split, MASK_DIR)
        for d in [self.pre_dir, self.post_dir, self.mask_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")

        self.pre_list = _list_images(self.pre_dir)
        if len(self.pre_list) == 0:
            raise RuntimeError(f"No images in {self.pre_dir}")

        self.pairs = []
        for p in self.pre_list:
            name = os.path.splitext(os.path.basename(p))[0]
            found_post = None
            found_mask = None
            for ext in IMG_EXTS:
                cp = os.path.join(self.post_dir, name + ext)
                cm = os.path.join(self.mask_dir, name + ext)
                if os.path.exists(cp) and found_post is None:
                    found_post = cp
                if os.path.exists(cm) and found_mask is None:
                    found_mask = cm
            if found_post and found_mask:
                self.pairs.append((p, found_post, found_mask))
        if len(self.pairs) == 0:
            raise RuntimeError("No matched triplets (t1/t2/label).")

        train_tf, val_tf = RSTransforms.BCDTransforms.get_transform_pipelines(self.args)
        self.transform = train_tf if split == 'train' else val_tf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_p, post_p, mask_p = self.pairs[idx]

        pre_img  = cv2.imread(pre_p)
        post_img = cv2.imread(post_p)
        mask_img = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        if pre_img is None:
            raise RuntimeError(f"[READ ERROR] PRE image failed: {pre_p}")
        if post_img is None:
            raise RuntimeError(f"[READ ERROR] POST image failed: {post_p}")
        if mask_img is None:
            raise RuntimeError(f"[READ ERROR] MASK failed: {mask_p}")

        pre  = cv2.cvtColor(pre_img,  cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        mask = mask_img
        mask = (mask > 0).astype(np.uint8)

        img_pair = np.concatenate([pre, post], axis=2)
        img_t, mask_t = self.transform(img_pair, mask)

        pre_t  = img_t[0:3, ...].contiguous()
        post_t = img_t[3:6, ...].contiguous()
        if mask_t.ndim == 2:
            mask_t = mask_t[None, ...]
        return {
            'pre': pre_t.float(), 'post': post_t.float(), 'mask': mask_t.float(),
            'pre_path': pre_p, 'post_path': post_p, 'mask_path': mask_p
        }


# ---------- utils ----------
def as_prob(t):
    if t.min() >= 0.0 and t.max() <= 1.0:
        return t.clamp(1e-6, 1-1e-6)
    return torch.sigmoid(t)

def bernoulli_logprob(logits, mask):
    log_p1 = -F.softplus(-logits)
    log_p0 = -logits + log_p1
    return mask * log_p1 + (1 - mask) * log_p0

def kl_bernoulli(logits_p, logits_q):
    p = as_prob(logits_p)
    q = as_prob(logits_q)
    kl = p * torch.log(p/q) + (1-p) * torch.log((1-p)/(1-q))
    return kl

def focal_loss(prob, target, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    bce = F.binary_cross_entropy(prob, target, reduction='none')
    pt = prob * target + (1 - prob) * (1 - target)
    focal = (alpha * target + (1 - alpha) * (1 - target)) * ((1 - pt).pow(gamma)) * bce
    return focal.mean()

def boundary_loss(prob, target):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=prob.dtype, device=prob.device)[None,None,...]
    sobel_y = sobel_x.transpose(2,3)
    tgt_edge = (F.conv2d(target, sobel_x, padding=1).abs() + F.conv2d(target, sobel_y, padding=1).abs())
    tgt_edge = (tgt_edge > 0).float().detach()
    prb_edge = (F.conv2d(prob, sobel_x, padding=1).abs() + F.conv2d(prob, sobel_y, padding=1).abs())
    prb_edge = torch.sigmoid(prb_edge)
    return F.binary_cross_entropy(prb_edge, tgt_edge)

def false_positive_penalty(prob, target, w=2.0):
    fp = prob * (1 - target)
    return (w * fp).mean()

def get_warmup_cosine_lr(step, total_steps):
    warmup = min(WARMUP_STEPS, max(1, total_steps // 2))
    if total_steps <= 1:
        return LR
    if step <= warmup:
        return LR * max(0.1, step / warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    cos_factor = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return LR * max(MIN_LR_FACTOR, cos_factor)

def iou_and_f1(pred_bin, target):
    inter = (pred_bin * target).sum(dim=(1,2,3))
    union = (pred_bin + target - pred_bin*target).sum(dim=(1,2,3)) + 1e-6
    iou = inter / union
    prec = inter / (pred_bin.sum(dim=(1,2,3)) + 1e-6)
    rec  = inter / (target.sum(dim=(1,2,3))   + 1e-6)
    f1   = 2*prec*rec / (prec+rec+1e-6)
    return iou.mean().item(), f1.mean().item(), prec.mean().item(), rec.mean().item()

def make_preference_masks(prob_policy, prob_ref, target, th=0.5, softness=6.0):
    p_mask = (prob_policy >= th).float()
    r_mask = (prob_ref    >= th).float()
    def iou(a,b):
        inter = (a*b).sum(dim=(1,2,3))
        union = (a + b - a*b).sum(dim=(1,2,3)) + 1e-6
        return inter/union
    iou_p = iou(p_mask, target)
    iou_r = iou(r_mask, target)
    prefer_policy = torch.sigmoid(softness * (iou_p - iou_r))[:,None,None,None]
    m_pos = prefer_policy * p_mask + (1 - prefer_policy) * r_mask
    m_neg = prefer_policy * r_mask + (1 - prefer_policy) * p_mask
    return m_pos.detach(), m_neg.detach()

def dpo_pairwise_loss(policy_logits, ref_logits, m_pos, m_neg, beta=BETA_DPO):
    lp_pos = bernoulli_logprob(policy_logits, m_pos).sum(dim=(1,2,3))
    lp_neg = bernoulli_logprob(policy_logits, m_neg).sum(dim=(1,2,3))
    lr_pos = bernoulli_logprob(ref_logits,    m_pos).sum(dim=(1,2,3))
    lr_neg = bernoulli_logprob(ref_logits,    m_neg).sum(dim=(1,2,3))
    d = (lp_pos - lp_neg) - (lr_pos - lr_neg)
    return F.softplus(-beta * d).mean()

class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.data.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

# ---------- build / eval ----------
def load_state_flexible(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=True)
    return model

def build_models(args):
    # 与 Trainer 需求对齐
    args.dataset = "WHU-CD"
    args.task = "BCD"
    args.num_perception_frame = 1
    args.in_height = 256
    args.in_width  = 256
    if not hasattr(args, "pretrained") or not args.pretrained:
        args.pretrained = "/fast/Wang/Change3D/X3D_L.pyth"

    # policy：待微调
    policy = Trainer(args).cuda()
    load_state_flexible(policy, args.ckpt)
    policy.train()

    # ref：冻结
    ref = Trainer(args).cuda()
    load_state_flexible(ref, args.ckpt)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    return policy, ref

def set_trainable_layers(model):
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "decoder" in n:
            p.requires_grad = True
        if BACKBONE_LAST_STAGE_TRAIN and ("backbone.stage4" in n or "backbone.layer4" in n):
            p.requires_grad = True

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, type=str)
    ap.add_argument('--save_path', required=True, type=str)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, type=str)
    ap.add_argument('--eval_only', action='store_true', help='仅评估，不训练；打印官方口径以及阈值扫')
    args = ap.parse_args()
    t_start = time.perf_counter()


    # ========= 在构造 Dataset/Model 之前先把关键字段补齐 =========
    args.dataset = "WHU-CD"
    args.task = "BCD"
    args.num_perception_frame = 1
    args.in_width  = 256
    args.in_height = 256
    if not hasattr(args, "pretrained") or not args.pretrained:
        args.pretrained = "/fast/Wang/Change3D/X3D_L.pyth"

    # 训练集（小而精）
    train_set = BCDPairDataset(args, args.data_root, 'train')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=WORKERS, pin_memory=True, drop_last=True)

    # 官方评估 loader
    val_loader  = build_eval_loader(args, 'val')
    test_loader = build_eval_loader(args, 'test')
    print(f"[INFO] val_official batches={len(val_loader)}  test_official batches={len(test_loader)}")

    # 模型
    policy, ref = build_models(args)
    set_trainable_layers(policy)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()),
                           lr=LR, weight_decay=1e-4)
    ema = ModelEMA(policy, EMA_DECAY)

    # 仅评估模式（官方口径 + 阈值扫）
    if args.eval_only:
        ema.apply_shadow(policy)
        iou_v, f1_v, p_v, r_v = evaluate_official(policy, val_loader)
        print(f"[OFFICIAL/VAL]  IoU={iou_v:.4f} F1={f1_v:.4f} P={p_v:.4f} R={r_v:.4f}")
        iou_t, f1_t, p_t, r_t = evaluate_official(policy, test_loader)
        print(f"[OFFICIAL/TEST] IoU={iou_t:.4f} F1={f1_t:.4f} P={p_t:.4f} R={r_t:.4f}")
        sweep_thresholds_official(policy, val_loader,  "val")
        sweep_thresholds_official(policy, test_loader, "test")
        ema.restore(policy)
        return

    # 训练前的“官方口径”基线
    iou0, f10, p0, r0 = evaluate_with_ema(policy, ema, val_loader)
    print(f"[BEFORE/official] val  IoU={iou0:.4f} F1={f10:.4f} P={p0:.4f} R={r0:.4f}")
    iout, f1t, pt, rt = evaluate_with_ema(policy, ema, test_loader)
    print(f"[BEFORE/official] test IoU={iout:.4f} F1={f1t:.4f} P={pt:.4f} R={rt:.4f}")

    best_f1 = -1.0
    total_steps = max(1, len(train_loader) * EPOCHS)
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        policy.train()
        for batch in train_loader:
            global_step += 1
            cur_lr = get_warmup_cosine_lr(global_step, total_steps)
            for g in opt.param_groups:
                g['lr'] = cur_lr

            pre  = batch['pre'].cuda(non_blocking=True)
            post = batch['post'].cuda(non_blocking=True)
            gt   = batch['mask'].cuda(non_blocking=True)

            logits_p = policy.update_bcd(pre, post)
            with torch.no_grad():
                logits_r = ref.update_bcd(pre, post)

            prob_p = as_prob(logits_p)
            prob_r = as_prob(logits_r)

            # 监督
            bce  = F.binary_cross_entropy(prob_p, gt)
            dice = 1.0 - (2*(prob_p*gt).sum(dim=(1,2,3)) / ((prob_p+gt).sum(dim=(1,2,3))+1e-6)).mean()
            foc  = focal_loss(prob_p, gt)
            sup_loss = (1 - FOCAL_WEIGHT) * (0.5*bce + 0.5*dice) + FOCAL_WEIGHT * foc

            # 边界 + 假阳
            bnd_loss = boundary_loss(prob_p, gt)
            fp_pen   = false_positive_penalty(prob_p, gt, w=2.0)

            # 偏好对（policy vs ref）
            m_pos, m_neg = make_preference_masks(prob_p.detach(), prob_r.detach(), gt, th=0.5)
            dpo_loss = dpo_pairwise_loss(logits_p, logits_r.detach(), m_pos, m_neg)

            # KL
            kl_loss = kl_bernoulli(logits_p, logits_r.detach()).mean()

            loss = (LAMBDA_SUP*sup_loss +
                    LAMBDA_BND*bnd_loss +
                    LAMBDA_FP*fp_pen +
                    LAMBDA_DPO*dpo_loss +
                    BETA_KL*kl_loss)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            ema.update(policy)

        iou, f1, prec, rec = evaluate_with_ema(policy, ema, val_loader)
        print(f"[Epoch {epoch:02d}/official] val IoU={iou:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            ema.apply_shadow(policy)
            torch.save(policy.state_dict(), args.save_path)
            ema.restore(policy)
            print(f"  -> saved to {args.save_path}")

    # 最终 test（官方口径）
    _ = load_state_flexible(policy, args.save_path)
    iou, f1, prec, rec = evaluate_official(policy, test_loader)
    print(f"[TEST/official] IoU={iou:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}")
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"[TIME] Total runtime = {elapsed/3600:.2f} hours ({elapsed:.1f} seconds)")


if __name__ == "__main__":
    main()
