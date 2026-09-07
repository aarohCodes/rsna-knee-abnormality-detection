"""Train one competition member and score it on the community's clean holdout.

This script IS the Phase-2 gate. It ends by printing our holdout macro AUC next to
0.9178 -- the score a single public DINOv3 member achieves on these exact 881
studies. That comparison decides whether the training track is worth five more
weeks:

    >= 0.90   the track is live; scale to 5-fold and multiple backbones
    <= 0.85   stop, revert to the robustness-only plan, keep the banked score

Our previous attempt scored 0.7735, so this is a rebuild rather than a tweak, and
the honest expectation is that it may fail. It should fail *fast* and *legibly*.

Example:
    python training/train_member.py \
        --cache /data/knee_cache --data-root /data/rsna \
        --silver data/interim/label_sources/steven/llm_labels_v4_blend.csv \
        --arch vit_small_patch14_dinov2.lvd142m --px 336 --epochs 8 --batch 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from member import (LABELS, KneeClassifier, StudyWindows, build_backbone,  # noqa: E402
                    load_targets, macro_auc)

PUBLIC_SINGLE_MEMBER = 0.9178   # what we must approach for this track to be worth it
OUR_PREVIOUS_ATTEMPT = 0.7735   # what we scored last time, on the same 881 studies


def collate(batch):
    xs, ys, ws = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.stack(ws)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    preds, trues = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(x)
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--silver", required=True)
    ap.add_argument("--arch", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--px", type=int, default=336)
    ap.add_argument("--k-train", type=int, default=24)
    ap.add_argument("--k-eval", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4, help="head LR")
    ap.add_argument("--encoder-lr", type=float, default=2e-5,
                    help="encoder LR; far below the head's on purpose (see below)")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/member")
    ap.add_argument("--limit", type=int, default=0, help="tiny subset, for smoke tests")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init: for smoke tests only, never for a real run")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={device} amp={amp_dtype if device.type=='cuda' else 'off'} arch={args.arch}", flush=True)

    tgt = load_targets(Path(args.data_root), Path(args.silver))
    cache = Path(args.cache)
    have = {p.stem for p in cache.glob("*.npz")}
    tgt = tgt[tgt["StudyInstanceUID"].isin(have)].reset_index(drop=True)
    if tgt.empty:
        print("ERROR: no cached studies match the label file. Is --cache correct?", file=sys.stderr)
        return 2

    tr = tgt[tgt["fold"] != args.holdout_fold].reset_index(drop=True)
    ho = tgt[tgt["fold"] == args.holdout_fold].reset_index(drop=True)
    if args.limit:
        tr, ho = tr.head(args.limit), ho.head(max(8, args.limit // 4))
    print(f"train={len(tr)}  holdout={len(ho)}  (gold in holdout: {int(ho['is_gold'].sum())})", flush=True)
    if len(ho) == 0:
        print("ERROR: holdout is empty", file=sys.stderr)
        return 2

    wts = lambda d: np.repeat(d["weight"].to_numpy()[:, None], len(LABELS), 1)
    ds_tr = StudyWindows(cache, tr["StudyInstanceUID"], tr[LABELS].to_numpy(), wts(tr),
                         k=args.k_train, px=args.px, train=True, seed=0)
    ds_ho = StudyWindows(cache, ho["StudyInstanceUID"], ho[LABELS].to_numpy(), wts(ho),
                         k=args.k_eval, px=args.px, train=False, seed=1)
    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=pin, drop_last=True, collate_fn=collate)
    dl_ho = DataLoader(ds_ho, batch_size=max(1, args.batch), shuffle=False,
                       num_workers=args.workers, pin_memory=pin, collate_fn=collate)

    backbone, feat_dim = build_backbone(args.arch, pretrained=not args.no_pretrained)
    model = KneeClassifier(backbone, feat_dim).to(device)

    # Discriminative LR: a self-supervised encoder has no vocabulary for MR signal,
    # so it must adapt -- but driving it at the head's LR while the head is still
    # random destroys the pretrained features.
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    enc_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": head_params, "lr": args.lr},
                             {"params": enc_params, "lr": args.encoder_lr}], weight_decay=0.01)
    steps = max(1, len(dl_tr)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.encoder_lr], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    best, history, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = n = 0
        for x, y, w in dl_tr:
            x, y, w = x.to(device, non_blocking=True), y.to(device), w.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = (lossf(model(x), y) * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach()) * len(x)
            n += len(x)
        p, t = evaluate(model, dl_ho, device, amp_dtype)
        auc = macro_auc(t, p)
        gold = ho["is_gold"].to_numpy()
        auc_gold = macro_auc(t[gold], p[gold]) if gold.sum() >= 8 else float("nan")
        history.append({"epoch": epoch, "loss": running / max(n, 1), "holdout_auc": auc,
                        "gold_auc": auc_gold, "secs": time.time() - t0})
        print(f"epoch {epoch}/{args.epochs}  loss={running/max(n,1):.4f}  "
              f"holdout_macro_auc={auc:.4f}  gold={auc_gold:.4f}  {time.time()-t0:.0f}s", flush=True)
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "arch": args.arch, "px": args.px,
                        "feat_dim": feat_dim, "holdout_auc": auc, "epoch": epoch},
                       out / "best.pt")
            pd.DataFrame(p, columns=LABELS).assign(
                StudyInstanceUID=ho["StudyInstanceUID"].values
            )[["StudyInstanceUID"] + LABELS].to_csv(out / "holdout_predictions.csv", index=False)

    (out / "history.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "best_holdout_auc": best}, indent=2, default=str))

    print("\n" + "=" * 68)
    print(f"  PHASE-2 GATE      our member          {best:.4f}")
    print(f"                    public single member {PUBLIC_SINGLE_MEMBER:.4f}")
    print(f"                    our previous attempt {OUR_PREVIOUS_ATTEMPT:.4f}")
    print("-" * 68)
    if best >= 0.90:
        print("  VERDICT: PASS -- scale to 5-fold and more backbones.")
    elif best >= 0.85:
        print("  VERDICT: MARGINAL -- one tuning round, then re-gate. Do not scale yet.")
    else:
        print("  VERDICT: FAIL -- stop the training track, keep the banked score.")
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
