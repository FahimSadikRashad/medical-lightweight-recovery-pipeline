#!/usr/bin/env bash
# Install everything needed to run the experiments.
#
#   bash scripts/install.sh
#
# medmnistc renders corruptions through ImageMagick via `wand`, so the system
# library must be installed BEFORE wand, and wand needs a force-reinstall
# afterwards to link against it. That ordering is the reason this is a script
# and not a plain requirements.txt.
set -e

PY="${PYTHON:-python3}"
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "== system: ImageMagick =="
if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq libmagickwand-dev imagemagick
else
    echo "!! no apt-get. Install ImageMagick manually, then re-run."
fi

echo "== python packages =="
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r requirements.txt
$PY -m pip install -q git+https://github.com/francescodisalvo05/medmnistc-api.git
# after ImageMagick exists, so wand links against it
$PY -m pip install -q --force-reinstall wand

echo "== check =="
$PY - <<'EOF'
import torch
print("torch:", torch.__version__)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "CPU ONLY (on Colab: Runtime > Change runtime type > GPU)")
from wand.version import MAGICK_VERSION_NUMBER
print("ImageMagick:", MAGICK_VERSION_NUMBER)
from robustmed.corruptions import names
print("corruptions:", names())
EOF

echo
echo "Done. Next: python scripts/download_data.py"
