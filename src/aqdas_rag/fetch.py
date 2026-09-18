"""Fetch the Kitáb-i-Aqdas fragments from the Bahá'í Reference Library.

The Reference Library serves each publication as numbered fragments under one
base path. We fetch them as raw bytes and verify the encoding before anything
downstream trusts the text: the corpus is full of transliterated names
(Bahá'u'lláh, Ḥuqúqu'lláh, Ṭihrán) whose diacritics are the first thing a bad
decode destroys, and a corrupted name fails silently at retrieval time rather
than loudly here.

We do not decide here which fragment holds which section. That is the parser's
job, derived from what each page actually contains.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://www.bahai.org/library/authoritative-texts/bahaullah/kitab-i-aqdas"

# The publication's fragments are sequential. We sweep a range rather than
# hardcoding the section->fragment mapping read off one page, so a renumbering
# upstream shows up as a parse result instead of a silent gap.
FRAGMENTS = range(1, 18)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aqdas-rag/0.1 (personal study project)"
POLITE_DELAY_S = 1.0


@dataclass(frozen=True)
class FetchResult:
    fragment: int
    path: Path
    n_bytes: int
    n_chars: int
    replacement_chars: int

    @property
    def ok(self) -> bool:
        return self.replacement_chars == 0 and self.n_bytes > 0


def fetch_fragment(n: int, out_dir: Path) -> FetchResult:
    """Download one fragment, verify it is strict UTF-8, write it verbatim."""
    url = f"{BASE}/{n}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw: bytes = resp.read()

    # Strict decode. A malformed byte raises here, while the file is still
    # untouched on disk -- rather than being written and discovered later.
    text = raw.decode("utf-8")

    out_path = out_dir / f"fragment-{n:02d}.html"
    # Write BYTES, not text: Path.write_text() translates "\n" to os.linesep,
    # which would rewrite the whole file to CRLF on Windows.
    out_path.write_bytes(raw)

    return FetchResult(
        fragment=n,
        path=out_path,
        n_bytes=len(raw),
        n_chars=len(text),
        replacement_chars=text.count("�"),
    )


def fetch_all(out_dir: Path) -> list[FetchResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []

    for n in FRAGMENTS:
        try:
            result = fetch_fragment(n, out_dir)
        except urllib.error.HTTPError as exc:
            print(f"  fragment {n:2d}  HTTP {exc.code} -- skipped")
            continue
        except UnicodeDecodeError as exc:
            print(f"  fragment {n:2d}  NOT VALID UTF-8: {exc}")
            continue

        flag = "ok" if result.ok else "SUSPECT"
        print(
            f"  fragment {n:2d}  {result.n_bytes:>7,} bytes  "
            f"{result.n_chars:>7,} chars  {flag}"
        )
        results.append(result)
        time.sleep(POLITE_DELAY_S)

    return results


def main() -> int:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    print(f"Fetching Kitáb-i-Aqdas fragments -> {out_dir}")
    results = fetch_all(out_dir)

    # Every figure below is counted from what actually landed on disk.
    total_bytes = sum(r.n_bytes for r in results)
    bad = [r for r in results if not r.ok]

    print()
    print(f"fragments fetched : {len(results)}")
    print(f"total bytes       : {total_bytes:,}")
    print(f"encoding failures : {len(bad)}")

    if bad:
        for r in bad:
            print(f"  !! {r.path.name}: {r.replacement_chars} replacement chars")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
