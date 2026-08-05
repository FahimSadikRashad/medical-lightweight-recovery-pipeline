"""Download the NLM Montgomery + Shenzhen chest X-ray sets and cache them at
IMAGE_SIZE as a single .npz.

These are the external chest X-rays for the Mode A transfer experiment: adult,
two institutions (Maryland and Shenzhen), and genuinely independent of Kermany
-- unlike the current Sub-Q2 set, which docs/PRESENTATION_FEEDBACK.md section 0
showed is PneumoniaMNIST's own test split at native resolution.

Why cache to .npz instead of keeping the files: the source PNGs are up to
4020x4892 and ~2.6 GB across 800 files, but the pipeline only ever consumes
IMAGE_SIZE. Decoding them takes minutes and would repeat every Colab session.
The cached array is ~10 MB, so it uploads in seconds and the decode happens once.

Resizing here is deliberately identical to data.Kermany -- open, convert("RGB"),
bare resize -- so external numbers stay comparable with the in-domain path.

    python scripts/prepare_nlm_tb.py
    python scripts/prepare_nlm_tb.py --keep-zips
    python scripts/prepare_nlm_tb.py --skip-download   # zips already on disk

Label convention: the filename suffix. MCUCXR_0001_0.png -> 0 = normal,
CHNCXR_0327_1.png -> 1 = abnormal. Note that "abnormal" here means
tuberculosis, NOT pneumonia -- see the note printed at the end.
"""
import argparse
import collections
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustmed import config  # noqa: E402

ZIPS = {
    "NLM-MontgomeryCXRSet.zip":
        "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
    "ChinaSet_AllFiles.zip":
        "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
}

# Published composition, from the NLM dataset descriptions. Only used to warn --
# if a release ever changes the filename label convention, silently mislabelled
# data is far worse than a failed run.
EXPECTED = {"montgomery": (80, 58), "shenzhen": (326, 336)}   # (normal, abnormal)


def download(url, dst):
    """curl with resume and retries. These are 0.6-2 GB over a research host
    that does drop connections, so -C - matters on a re-run."""
    if dst.exists():
        print(f"  have {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  fetching {dst.name} ...", flush=True)
    subprocess.run([
        "curl", "-L", "--fail", "-C", "-",
        "--retry", "10", "--retry-delay", "10", "--retry-all-errors",
        "--connect-timeout", "60",
        "-o", str(dst), url,
    ], check=True)


def extract(zip_path, out_dir):
    """Only CXR_png/*.png.

    ManualMask/ holds lung segmentation masks under the SAME filenames as the
    X-rays, so a broad extract plus a broad glob would yield three entries per
    Montgomery image, two of them binary masks labelled as chest X-rays.
    ClinicalReadings/ is 800 text files we never read.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if "/CXR_png/" in m and m.lower().endswith(".png")]
        zf.extractall(out_dir, members=members)
    print(f"  extracted {len(members)} images from {zip_path.name}")


def collect(out_dir):
    items = []
    for p in sorted(out_dir.rglob("*/CXR_png/*.png")):
        if "__MACOSX" in p.parts or p.stem[-1] not in "01":
            continue
        src = "montgomery" if "MontgomerySet" in p.parts else "shenzhen"
        items.append((p, int(p.stem[-1]), src))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external",
                    help="where zips and the .npz go (default: data/external)")
    ap.add_argument("--size", type=int, default=config.IMAGE_SIZE,
                    help=f"cache resolution (default: config.IMAGE_SIZE={config.IMAGE_SIZE})")
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep the ~2.6 GB of source zips after caching")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"== download -> {raw_dir} ==")
    for name, url in ZIPS.items():
        if not args.skip_download:
            download(url, raw_dir / name)
        extract(raw_dir / name, raw_dir)

    items = collect(raw_dir)
    if not items:
        raise SystemExit(f"no images found under {raw_dir} -- extraction failed?")

    print(f"\n== decode + resize to {args.size}x{args.size} ==")
    imgs, labels, sources, modes = [], [], [], collections.Counter()
    for i, (path, label, src) in enumerate(items):
        with Image.open(path) as im:
            modes[im.mode] += 1
            # identical to data.Kermany: bare resize, so PIL's default
            # (BICUBIC for RGB) applies in both paths
            arr = np.array(im.convert("RGB").resize((args.size,) * 2))
        imgs.append(arr)
        labels.append(label)
        sources.append(src)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)

    imgs = np.stack(imgs).astype("uint8")
    labels = np.array(labels, dtype="int64")
    sources = np.array(sources)

    npz = out_dir / f"nlm_tb_{args.size}.npz"
    np.savez_compressed(npz, images=imgs, labels=labels, sources=sources)

    # --- report -----------------------------------------------------------
    print(f"\nsource modes: {dict(modes)}")
    if any(m not in ("L", "RGB") for m in modes):
        print("  !! unexpected PIL mode. 16-bit ('I;16') would make convert('RGB')\n"
              "     truncate and produce near-black images -- inspect before using.")
    print(f"mean intensity: {imgs.mean():.1f}/255  (a value near 0 means the "
          f"decode went wrong)")

    print(f"\n{'source':12s} {'normal':>7s} {'abnormal':>9s}")
    ok = True
    for src, (exp_n, exp_a) in EXPECTED.items():
        n = int(((sources == src) & (labels == 0)).sum())
        a = int(((sources == src) & (labels == 1)).sum())
        ok &= (n, a) == (exp_n, exp_a)
        flag = "" if (n, a) == (exp_n, exp_a) else f"   <-- expected {exp_n}/{exp_a}"
        print(f"{src:12s} {n:7d} {a:9d}{flag}")
    print(f"{'total':12s} {int((labels == 0).sum()):7d} {int((labels == 1).sum()):9d}")

    if not ok:
        print("\n!! counts differ from the published composition -- the filename\n"
              "   label convention may have changed. Resolve before using.")

    if not args.keep_zips:
        for name in ZIPS:
            (raw_dir / name).unlink(missing_ok=True)
        print(f"\nremoved source zips (--keep-zips to retain)")

    print(f"\nsaved -> {npz}  ({npz.stat().st_size / 1e6:.1f} MB)")
    print("\nUpload that one file to Colab; scripts/10_external_transfer.py "
          "reads it directly.")
    print("\nReminder: label 1 is TUBERCULOSIS-abnormal, not pneumonia. Transfer "
          "results\nfrom this set are abnormality detection, which is weaker than "
          "the\nlabel-matched comparison NIH ChestX-ray14 would give.")


if __name__ == "__main__":
    main()
