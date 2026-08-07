"""Fetch the non-MedMNIST corpora and unpack them where the loaders expect.

    python scripts/download_external.py --corpus tb_cxr
    python scripts/download_external.py --corpus busi --url <mirror>
    python scripts/download_external.py --corpus tb_cxr --archive /path/to/set.zip

On Kaggle this needs Settings -> Internet: On, or every mirror will look dead.

Why each archive carries several URLs. NLM serves the TB collection from two
official hosts and they do not behave the same way. Checked live:

    openi.nlm.nih.gov          200 OK, no User-Agent needed
    data.lhncbc.nlm.nih.gov    403 even WITH a browser User-Agent

So openi is tried first and lhncbc is the fallback, not the other way round. A
browser UA is still sent because it costs nothing and some hosts do gate on it,
but it is not what makes this work -- the mirror order is.

SIZE WARNING. These are full-resolution radiographs, not thumbnails:
Montgomery is 617 MB and Shenzhen is 3.8 GB, so ~4.4 GB down and roughly the
same again once extracted. On Kaggle that is a real fraction of the 20 GB
working directory -- pass --prune-archives to delete each zip after it extracts.

Needs Settings -> Internet: On when running on Kaggle, or every mirror looks dead.

Nothing here is destructive by default: an archive already on disk is not
re-fetched, and extraction is skipped when the destination is already populated.
"""
import argparse
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustmed import external  # noqa: E402

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")

# (archive name, [mirrors in preference order]). Both hosts are official NLM.
SOURCES = {
    "tb_cxr": [
        ("NLM-MontgomeryCXRSet.zip", [
            "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
            "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
            "/Montgomery-County-CXR-Set/MontgomerySet.zip",
        ]),
        ("ChinaSet_AllFiles.zip", [
            "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
            "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
            "/Shenzhen-Hospital-CXR-Set/ChinaSet_AllFiles.zip",
        ]),
    ],
    "busi": [],   # no stable direct link; pass --url or --archive
}

DEST = {"tb_cxr": "data/external/tb_cxr", "busi": "data/external/busi"}


def fetch(urls, path):
    """Try each mirror in order. Returns the path, or None if all failed."""
    if os.path.exists(path) and zipfile.is_zipfile(path):
        print(f"  have {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for url in urls:
        print(f"  GET {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r      {100*got/total:5.1f}%  {got/1e6:7.1f} /"
                              f" {total/1e6:.1f} MB", end="", flush=True)
                    else:
                        print(f"\r      {got/1e6:7.1f} MB", end="", flush=True)
            print()
        except Exception as exc:
            code = getattr(exc, "code", None)
            hint = {403: " -- server refused the client, not a bad path",
                    404: " -- moved or removed"}.get(code, "")
            print(f"      failed: {type(exc).__name__} {code or exc}{hint}")
            if os.path.exists(path):
                os.remove(path)
            continue

        # A redirect to an HTML error page saved under a .zip name is the classic
        # failure, and it only surfaces later as a confusing extract error.
        if not zipfile.is_zipfile(path):
            print(f"      got {os.path.getsize(path)/1e6:.2f} MB but it is not a "
                  f"zip -- discarding")
            os.remove(path)
            continue
        print(f"  saved {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e6:.1f} MB)")
        return path
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(SOURCES))
    ap.add_argument("--url", action="append", default=[],
                    help="extra mirror, tried before the built-in ones")
    ap.add_argument("--archive", action="append", default=[],
                    help="archive already on disk; skips downloading it")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--prune-archives", action="store_true",
                    help="delete each zip once it has extracted (~4.4 GB saved; "
                         "for Kaggle's 20 GB working directory)")
    args = ap.parse_args()

    dest = args.dest or DEST[args.corpus]
    os.makedirs(dest, exist_ok=True)

    jobs = [(os.path.basename(u.split("?")[0]), [u]) for u in args.url]
    jobs += SOURCES[args.corpus]

    archives, failed = [], []
    for local in args.archive:
        if not zipfile.is_zipfile(local):
            raise SystemExit(f"{local} is not a zip archive")
        print(f"  using {local}")
        archives.append(local)

    for name, urls in jobs:
        path = fetch(urls, os.path.join(dest, "_archives", name))
        (archives if path else failed).append(path or name)

    for name in failed:
        print(f"\n  !! {name} unavailable from every mirror.\n"
              f"     Download it in a browser, then re-run with "
              f"--archive /path/to/{name}")

    if not archives:
        raise SystemExit(f"\nnothing to extract into {dest}")

    for path in archives:
        sub = os.path.join(dest, os.path.splitext(os.path.basename(path))[0])
        print(f"  extracting {os.path.basename(path)} -> {sub}")
        external._unzip(path, sub)
        if args.prune_archives and path.startswith(dest):
            # Only ever remove archives we downloaded into dest -- never one the
            # user passed in with --archive.
            os.remove(path)
            print(f"  removed {os.path.basename(path)}")

    print("\nverifying with the loader ...")
    tr, va, te, info = external.LOADERS[args.corpus](dest)
    reg = external.SUGGESTED_REGISTRY[args.corpus]
    print(f"\nready:\n"
          f"  python scripts/01_baseline1_frozen.py  "
          f"--dataset {args.corpus} --registry {reg}\n"
          f"  python scripts/13_module_comparison.py "
          f"--dataset {args.corpus} --registry {reg} --published "
          f"--arms identity unsharp convae --seeds 0 1 2\n"
          f"  python scripts/15_qualitative.py       "
          f"--dataset {args.corpus} --registry {reg}")
