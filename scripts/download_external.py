"""Fetch the non-MedMNIST corpora and unpack them where the loaders expect.

    python scripts/download_external.py --corpus tb_cxr
    python scripts/download_external.py --corpus busi --url <mirror>

tb_cxr (Montgomery + Shenzhen) is served openly by the NLM with no registration
and no data-use agreement, which is why it is the first external corpus: two
downloads and it runs.

BUSI has no single canonical direct URL that stays stable, so it takes --url or
a manually placed archive. That is stated rather than papered over with a link
that may rot -- a broken download in a paper's repo is worse than an explicit
instruction.

Nothing here is destructive: existing archives are not re-downloaded, and
extraction is skipped when the destination already has files.
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustmed import external  # noqa: E402

# NLM's open TB collection. Verify these resolve before a long run -- print the
# response size first; a redirect to an HTML error page is the usual failure and
# it produces a "zip" that will not open.
SOURCES = {
    "tb_cxr": [
        ("https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
         "/Montgomery-County-CXR-Set/MontgomerySet.zip", "MontgomerySet.zip"),
        ("https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
         "/Shenzhen-Hospital-CXR-Set/ChinaSet_AllFiles.zip", "ChinaSet_AllFiles.zip"),
    ],
    "busi": [],   # supply --url; see the module docstring
}

DEST = {"tb_cxr": "data/external/tb_cxr", "busi": "data/external/busi"}


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"  have {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return path
    print(f"  downloading {url}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def hook(blocks, size, total):
        if total > 0:
            pct = min(100, 100 * blocks * size / total)
            print(f"\r    {pct:5.1f}%  {blocks*size/1e6:7.1f} / {total/1e6:.1f} MB",
                  end="", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=hook)
    print()
    mb = os.path.getsize(path) / 1e6
    if mb < 1:
        raise SystemExit(
            f"  {path} is only {mb:.2f} MB -- almost certainly an error page, not "
            f"an archive. Open the URL in a browser to check it still resolves.")
    print(f"  saved {path} ({mb:.1f} MB)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(SOURCES))
    ap.add_argument("--url", action="append", default=[],
                    help="extra archive URL (required for busi)")
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()

    dest = args.dest or DEST[args.corpus]
    urls = [(u, os.path.basename(u.split("?")[0])) for u in args.url]
    urls += SOURCES[args.corpus]
    if not urls:
        raise SystemExit(
            f"no source for {args.corpus!r}. Pass --url, or download the archive "
            f"manually and unpack it under {dest}/.")

    os.makedirs(dest, exist_ok=True)
    for url, name in urls:
        path = fetch(url, os.path.join(dest, "_archives", name))
        print(f"  extracting -> {dest}")
        external._unzip(path, dest)

    print(f"\nverifying with the loader ...")
    tr, va, te, info = external.LOADERS[args.corpus](dest)
    reg = external.SUGGESTED_REGISTRY[args.corpus]
    print(f"\nready. Run with:\n"
          f"  python scripts/01_baseline1_frozen.py "
          f"--dataset {args.corpus} --registry {reg}\n"
          f"  python scripts/13_module_comparison.py --published "
          f"--dataset {args.corpus} --registry {reg} --arms identity unsharp convae\n"
          f"  python scripts/15_qualitative.py --dataset {args.corpus} --registry {reg}")
