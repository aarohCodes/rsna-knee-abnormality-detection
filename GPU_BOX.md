# Start here on the GPU box

Everything is copy-paste. Do the steps in order; **step 1 decides the rest**, so
don't skip ahead to downloading anything.

Goal: **top 10 on the private leaderboard, target 0.948.** We are at 0.938. This
machine exists to produce the +0.010 that the public ecosystem cannot.

---

## Step 0 — get the repo and credentials onto the box

```bash
# clone (private repo -> use a GitHub personal access token as the password)
git clone https://github.com/aarohCodes/rsna-knee-abnormality-detection.git
cd rsna-knee-abnormality-detection

# Kaggle credentials, copied from the Windows box
#   source: C:\Users\aaroh\.kaggle\kaggle.json
mkdir -p ~/.kaggle
#   ...scp or paste the file to ~/.kaggle/kaggle.json, then:
chmod 600 ~/.kaggle/kaggle.json
```

From Windows, the scp one-liner is:

```powershell
scp $HOME\.kaggle\kaggle.json justingpu.utdallas.edu:~/.kaggle/
```

---

## Step 1 — RECON. Run this first and send me the output.

```bash
echo "=== GPU ==="; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo "=== who else is on it ==="; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "=== CPU / RAM ==="; nproc; free -g | head -2
echo "=== DISK ==="; df -h ~ /tmp /scratch /data /local 2>/dev/null | grep -v '^df:'
```

**Why this gates everything:**

| what it tells us | why it matters |
|---|---|
| GPU model + VRAM | sets `--batch`; 24GB → 8, 16GB → 4, 48GB+ → 16 |
| how many GPUs | if >1, we pin to one with `CUDA_VISIBLE_DEVICES=0` and leave the rest free |
| whether others are running | decides whether to `nice` the job and cap VRAM |
| free disk | **decides Step 2**: 500GB free → download raw; less → build the cache on Kaggle |

---

## Step 2 — environment

```bash
bash training/setup_linux.sh /path/for/data
source .venv-linux/bin/activate
```

It refuses to continue if `nvidia-smi` fails or torch can't see CUDA — without a
working GPU this box is no better than Kaggle's free T4s. It also pulls the silver
labels from Kaggle (they are **not** in git, because `/data/` is gitignored).

`/path/for/data` is only an example. Nothing in the repo assumes a username or a
drive; every path is an argument or derived from the script's location.

---

## Step 3 — get the cache. Two routes; Step 1 decides which.

The raw corpus is ~500GB–1TB, but **nothing trains on raw DICOM**. Everything
decodes once to a fixed uint8 cache:

| cache | slices | size |
|---|---|---|
| `raptor` (public recipe) | 64 | 27.7 GB |
| `lean` (efficiency member) | 32 | 13.5 GB |

**Route A — plenty of disk (600GB+ free).** `setup_linux.sh` already downloaded
the corpus, so just build:

```bash
python training/prep_cache.py --data-root /path/for/data --out /path/for/data/cache --limit 20   # smoke test, ~1 min
python training/prep_cache.py --data-root /path/for/data --out /path/for/data/cache             # full, a few hours, resumable
```

**Route B — limited disk.** Build the cache on a **Kaggle CPU notebook** (the data
is already mounted there, and CPU notebooks **do not touch the 30h GPU quota**),
publish it as a Dataset, then download only the cache:

```bash
# on Kaggle, in a CPU notebook, with the competition attached:
python prep_cache.py --data-root /kaggle/input/rsna-knee-abnormality-detection \
                     --out /kaggle/working/cache --shard 1/2
# second run for --shard 2/2, since Kaggle caps notebook output near 20GB
# (the lean preset fits in one run)

# then on this box:
kaggle datasets download -d <your-username>/<cache-dataset> -p /path/for/data --unzip
```

---

## Step 4 — the first member, and the gate

```bash
python training/train_member.py \
    --cache /path/for/data/cache --data-root /path/for/data \
    --silver data/interim/label_sources/steven/llm_labels_v4_blend.csv \
    --arch vit_base_patch14_dinov2.lvd142m --px 336 --epochs 16 --batch 8
```

Use `tmux` so a dropped connection doesn't kill a 3-hour run:

```bash
tmux new -s train      # start; Ctrl-B then D to detach
tmux attach -t train   # reattach later, from any login
```

If the box is shared, be a good citizen:

```bash
CUDA_VISIBLE_DEVICES=0 nice -n 10 python training/train_member.py ... --batch 4 --workers 4
```

**Then run the actual gate** — standalone AUC is *not* the criterion:

```bash
python training/infer_member.py --checkpoint runs/member/best.pt \
    --data-root /path/for/data --split train --holdout-fold 0 --out member_holdout.csv

python training/blend_member.py choose-weight \
    --base  data/interim/kernel_output_dino_eval/submission_e10_v2.csv \
    --member member_holdout.csv \
    --labels data/interim/label_sources/steven/llm_labels_v4_blend.csv
```

| honest delta | verdict |
|---|---|
| **> +0.002** | ship it — train the remaining 3 members |
| ≤ +0.002 | stop; the training track does not pay |

**Why the blend delta and not the score:** a member scoring **0.8963** — *below*
the 0.9202 base — still added **+0.0026**, because its rank correlation was only
0.843. Decorrelation is what makes a member valuable, not raw accuracy.

---

## Step 5 — the remaining three members

Different architecture **families**, because that is what decorrelates:

| # | command fragment | est. inference on 1,322 studies |
|---|---|---|
| 1 | `--arch vit_base_patch14_dinov2.lvd142m` | ~22 min |
| 2 | `--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --px 384` | ~30 min |
| 3 | `--arch convnext_small.in12k_ft_in1k` | ~20 min |
| 4 | `--arch vit_base_patch14_dinov2.lvd142m --within-slot` + a `balanced` cache | ~22 min |

**Do not use ViT-giant.** Its ~4.4h of inference would consume the entire runtime
headroom in the submission kernel, and four decorrelated members beat one large one.

All three backbones were verified locally with real pretrained forward passes.

---

## Continuing the conversation on this box

You **cannot** resume the Windows chat here — Claude Code session state is local to
the machine. Two options:

1. **Run the commands yourself and paste output back into the existing chat.** No
   install, keeps all the accumulated context.
2. **Install Claude Code here** and let it read its way in:
   ```bash
   npm install -g @anthropic-ai/claude-code    # or: curl -fsSL https://claude.ai/install.sh | bash
   cd ~/rsna-knee-abnormality-detection && claude
   ```
   Then tell it: **"read STATUS.md and GPU_BOX.md"**. `STATUS.md` opens with the
   goal, the arithmetic, the six-step plan, and every measurement behind it —
   including the corrections where evidence overturned an earlier assumption.
