"""One-off helper: pull a handful of real train_series/ DICOM files for local pipeline
testing. Not part of the reusable package -- the file listing API has no prefix/path
filter, so this must page past the (huge) test_series/ listing first. Real training
happens on Kaggle's GPU against the full mounted dataset; this is only for validating
DICOM I/O locally before pushing code to a kernel.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
N_STUDIES = 10
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "train_series"


def kaggle_auth_header() -> dict:
    cfg = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    token = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def list_page(headers: dict, page_token: str | None) -> dict:
    url = f"https://www.kaggle.com/api/v1/competitions/data/list/{COMPETITION}?pageSize=200"
    if page_token:
        url += f"&pageToken={page_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def download_file(headers: dict, file_name: str, dest: Path, max_retries: int = 6) -> None:
    # Nested paths need their slashes percent-encoded (%2F) -- the download endpoint's
    # routing 404s on a literal "/" even though the list endpoint returns plain names.
    encoded = urllib.parse.quote(file_name, safe="")
    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/{encoded}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as out:
                out.write(r.read())
            return
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2**attempt  # 1, 2, 4, 8, 16, 32s
                print(f"  429 rate limited, backing off {wait}s...", file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise


def main() -> None:
    headers = kaggle_auth_header()
    token = None
    page = 0
    train_series_files: list[str] = []
    studies_seen: set[str] = set()

    t0 = time.time()
    while True:
        page += 1
        data = list_page(headers, token)
        names = [f["name"] for f in data.get("files", [])]
        hit_train_series = False
        for n in names:
            if n.startswith("train_series/") and n.endswith(".dcm"):
                hit_train_series = True
                study_id = n.split("/")[1]
                if study_id in studies_seen or len(studies_seen) < N_STUDIES:
                    studies_seen.add(study_id)
                    train_series_files.append(n)
        if page % 25 == 0 or hit_train_series:
            print(
                f"page {page}, elapsed {time.time()-t0:.0f}s, "
                f"studies collected so far: {len(studies_seen)}, "
                f"last name seen: {names[-1] if names else None}",
                file=sys.stderr,
                flush=True,
            )
        token = data.get("nextPageToken")
        if len(studies_seen) >= N_STUDIES and hit_train_series and len(names) < 200:
            # Started collecting and this page had a partial/last batch -- one more
            # page to be safe, then stop.
            pass
        if len(studies_seen) >= N_STUDIES:
            # Grab a couple more pages to make sure we captured full series for the
            # last study or two, then stop.
            for _ in range(3):
                if not token:
                    break
                page += 1
                data = list_page(headers, token)
                for n in [f["name"] for f in data.get("files", [])]:
                    if (
                        n.startswith("train_series/")
                        and n.endswith(".dcm")
                        and n.split("/")[1] in studies_seen
                    ):
                        train_series_files.append(n)
                token = data.get("nextPageToken")
            break
        if not token:
            print("Ran out of pages before collecting enough studies.", file=sys.stderr)
            break

    print(f"Collected {len(train_series_files)} files across {len(studies_seen)} studies "
          f"in {page} pages ({time.time()-t0:.0f}s). Downloading...", file=sys.stderr)

    total_bytes = 0
    for i, name in enumerate(train_series_files):
        rel = "/".join(name.split("/")[1:])  # strip leading "train_series/"
        dest = OUT_DIR / rel
        if dest.exists():
            continue
        download_file(headers, name, dest)
        total_bytes += dest.stat().st_size
        time.sleep(0.1)  # small pacing delay to avoid re-triggering the rate limit
        if (i + 1) % 50 == 0:
            print(f"  downloaded {i+1}/{len(train_series_files)} files "
                  f"({total_bytes / 1024**2:.0f} MB)", file=sys.stderr, flush=True)

    print(f"Done. {len(train_series_files)} files, {total_bytes / 1024**2:.0f} MB, "
          f"studies: {sorted(studies_seen)}")


if __name__ == "__main__":
    main()
