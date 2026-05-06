# python .\summarize_tpa.py --input_dir "D:\path\to\your\folder" --output_name "my_summary.csv"
#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import List, Tuple

def read_spectrum(fp: Path) -> Tuple[List[float], List[float]]:
    """
    Reads a 2-column text/CSV (header tolerated).
    Returns (wavelengths, sigmas) as floats, sorted by wavelength.
    """
    wavelengths, sigmas = [], []
    with fp.open("r", newline="") as f:
        text = f.read()
        if not text.strip():
            return [], []
        f.seek(0)
        # Robust CSV dialect sniffing
        try:
            dialect = csv.Sniffer().sniff(text[:2048])
        except Exception:
            dialect = csv.get_dialect("excel")
        reader = csv.reader(f, dialect)
        rows = list(reader)
        if not rows:
            return [], []

        # Try to detect header
        has_header = False
        try:
            has_header = csv.Sniffer().has_header(text[:2048])
        except Exception:
            pass

        if has_header:
            header = [h.strip() for h in rows[0]]
            data_rows = rows[1:]
            wl_idx = sg_idx = None
            for i, name in enumerate(header):
                n = name.lower()
                if wl_idx is None and "wave" in n:
                    wl_idx = i
                if sg_idx is None and ("sigma" in n or "gm" in n):
                    sg_idx = i
            if wl_idx is None or sg_idx is None:
                wl_idx, sg_idx = (0, 1) if len(header) > 1 else (0, 0)
        else:
            data_rows = rows
            wl_idx, sg_idx = 0, 1

        for r in data_rows:
            if len(r) < 2:
                continue
            try:
                wl = float(str(r[wl_idx]).strip())
                sg = float(str(r[sg_idx]).strip())
            except Exception:
                continue
            wavelengths.append(wl)
            sigmas.append(sg)

    pairs = sorted(zip(wavelengths, sigmas), key=lambda x: x[0])
    if not pairs:
        return [], []
    wls, sgs = zip(*pairs)
    return list(wls), list(sgs)

def most_common_step(wls: List[float]) -> float:
    if len(wls) < 2:
        return 0.0
    diffs = [round(wls[i+1]-wls[i], 6) for i in range(len(wls)-1) if wls[i+1] > wls[i]]
    if not diffs:
        return 0.0
    step, _ = Counter(diffs).most_common(1)[0]
    return step

def compress_to_ranges(values: List[float], step: float) -> List[Tuple[float, float]]:
    """Compress sorted unique values into contiguous ranges using inferred step."""
    if not values:
        return []
    vals = sorted(set(values))
    if step <= 0:
        return [(v, v) for v in vals]
    ranges = []
    start = prev = vals[0]
    tol = max(1e-6, step * 0.05)  # 5% tolerance
    for v in vals[1:]:
        if abs((v - prev) - step) <= tol:
            prev = v
        else:
            ranges.append((start, prev))
            start = prev = v
    ranges.append((start, prev))
    return ranges

def format_nm(x: float) -> str:
    return f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:g}"

def format_ranges(ranges: List[Tuple[float, float]]) -> str:
    """Format like '600-640; 660; 700-720'."""
    parts = []
    for a, b in ranges:
        if abs(a - b) < 1e-9:
            parts.append(format_nm(a))
        else:
            parts.append(f"{format_nm(a)}-{format_nm(b)}")
    return "; ".join(parts)

def top_k_indices(values: List[float], k: int) -> List[int]:
    """Indices of k largest values (stable; tie-break by smaller index)."""
    ranked = sorted(enumerate(values), key=lambda x: (-x[1], x[0]))
    return [i for i, _ in ranked[:k]]

def pick_max_with_tie_break(wls: List[float], sgs: List[float]) -> Tuple[float, float]:
    """Return (max_sigma, wavelength_of_max) with tie-break to smallest wavelength."""
    if not sgs:
        return float("nan"), float("nan")
    max_val = max(sgs)
    idxs = [i for i, v in enumerate(sgs) if v == max_val]
    wl = min(wls[i] for i in idxs)
    return max_val, wl

def process_file(fp: Path, threshold: float = 20.0) -> Tuple[str, float, float]:
    """
    Apply rules:
      - If max(sigma) <= threshold: pick top-5 sigmas → compress wavelengths to ranges.
      - Else (exists sigma > threshold): pick all wavelengths with sigma > threshold → compress.
    Return (range_str, max_sigma, max_sigma_wavelength).
    """
    wls, sgs = read_spectrum(fp)
    if not wls:
        return "", float("nan"), float("nan")

    step = most_common_step(wls)
    max_sigma, max_sigma_wl = pick_max_with_tie_break(wls, sgs)

    if max_sigma <= threshold:
        idxs = top_k_indices(sgs, 5)
        sel_wls = [wls[i] for i in idxs]
    else:
        sel_wls = [wl for wl, sg in zip(wls, sgs) if sg > threshold]

    ranges = compress_to_ranges(sel_wls, step)
    return format_ranges(ranges), max_sigma, max_sigma_wl

def main():
    ap = argparse.ArgumentParser(description="Create ONE master CSV summarizing absorption ranges for all spectra.")
    ap.add_argument("--input_dir", required=True, help="Directory containing TXT/CSV spectra")
    ap.add_argument("--pattern", default="*.txt", help='Glob for inputs, e.g. "*.txt" or "*.csv"')
    ap.add_argument("--output_dir", default=None, help="Directory to write the master CSV (default: input_dir)")
    ap.add_argument("--output_name", default="absorption_summary_all.csv", help="Output CSV filename")
    ap.add_argument("--threshold", type=float, default=20.0, help="Sigma threshold (default: 20)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        alt = "*.csv" if args.pattern == "*.txt" else "*.txt"
        files = sorted(in_dir.glob(alt))
    if not files:
        raise SystemExit(f"No files found in {in_dir} matching {args.pattern!r} or its alternate.")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / args.output_name

    with out_fp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "wavelength_range", "max_sigma", "max_sigma_wavelength"])
        for fp in files:
            range_str, max_sigma, max_sigma_wl = process_file(fp, threshold=args.threshold)
            writer.writerow([fp.name, range_str, f"{max_sigma:.6g}", format_nm(max_sigma_wl)])

    print(f"Wrote master CSV: {out_fp}")

if __name__ == "__main__":
    main()

