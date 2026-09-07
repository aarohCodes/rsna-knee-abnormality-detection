#!/usr/bin/env bash
# Set up the Linux training box and pull the corpus.
#
# Run from the repo root:   bash training/setup_linux.sh /data/rsna
#
# The single argument is where the ~500GB-1TB DICOM corpus will live. Everything
# else (venv, cache) is derived from it. Safe to re-run: every step is idempotent.
set -euo pipefail

DATA_ROOT="${1:?usage: bash training/setup_linux.sh /path/for/data}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv-linux"

echo "== 0. environment check =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! nvidia-smi failed. Without a working NVIDIA driver this box is no better"
  echo "!! than Kaggle's T4s and the training track is not worth starting."; exit 1; }
echo "CPU cores: $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
echo "Disk at $(dirname "$DATA_ROOT"):"; df -h "$(dirname "$DATA_ROOT")" | tail -1

echo
echo "== 1. python env =="
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -qU pip wheel
# cu124 wheels cover every current NVIDIA consumer and datacentre card
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -q timm pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg \
                        python-gdcm opencv-python-headless pandas numpy scipy kaggle
python - <<'EOF'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB  "
          f"bf16={torch.cuda.is_bf16_supported()}")
else:
    raise SystemExit("CUDA not visible to torch -- fix the driver/toolkit before continuing")
EOF

echo
echo "== 2. kaggle credentials =="
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "!! Missing ~/.kaggle/kaggle.json."
  echo "!! Copy it from the Windows box (C:\\Users\\aaroh\\.kaggle\\kaggle.json), then:"
  echo "!!   mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json"
kaggle competitions list -s rsna-knee-abnormality-detection >/dev/null && echo "kaggle auth OK"

echo
echo "== 3. download the corpus (long; safe to resume) =="
mkdir -p "$DATA_ROOT"
if [ ! -f "$DATA_ROOT/train.csv" ]; then
  kaggle competitions download -c rsna-knee-abnormality-detection -p "$DATA_ROOT"
  echo "unzipping (this also takes a while)..."
  unzip -q -o "$DATA_ROOT/rsna-knee-abnormality-detection.zip" -d "$DATA_ROOT"
  # keep the zip until the cache is built and verified, then delete it by hand
else
  echo "already present, skipping"
fi
ls -1 "$DATA_ROOT" | head
echo "train studies on disk: $(ls -1 "$DATA_ROOT/train_series" 2>/dev/null | wc -l)  (expect 4407)"

cat <<EOF

== next ==
  source "$VENV/bin/activate"
  # 1. smoke-test the cache builder on 20 studies (~1 min)
  python training/prep_cache.py --data-root "$DATA_ROOT" --out "$DATA_ROOT/cache" --limit 20
  # 2. build the full cache (~28 GB; hours, parallel over cores)
  python training/prep_cache.py --data-root "$DATA_ROOT" --out "$DATA_ROOT/cache"
  # 3. train one member -- this prints the Phase-2 gate verdict
  python training/train_member.py --cache "$DATA_ROOT/cache" --data-root "$DATA_ROOT" \\
      --silver data/interim/label_sources/steven/llm_labels_v4_blend.csv \\
      --arch vit_small_patch14_dinov2.lvd142m --px 336 --epochs 8 --batch 8
EOF
