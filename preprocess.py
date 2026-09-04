#!/usr/bin/env python3
"""preprocess.py — Section 8 (steps 1–8) applied to every window in index.csv.

For each window listed in <dataset-dir>/index.csv, this script:

  1. Loads the correct ECG / IMU slice from the raw source CSV
  2. Runs quality checks (NaN fraction, flatline, ECG_EMG_Status1 flag)
  3. Bandpass filters ECG (0.5–40 Hz, 4th order, zero-phase)
  4. Notch filters at 50 Hz (Q = 30) — power-line hum, dataset from Germany
  5. Extra high-pass at 0.67 Hz — residual respiratory baseline wander
  6. Per-window z-score normalization
  7. Detects R-peaks with neurokit2 (Pan-Tompkins under the hood)
  8. Segments beats: 200 ms before to 400 ms after R-peak = 307 samples

Successful windows are saved as compressed .npz files at:

  <dataset-dir>/processed/<global_subject_id>/<sample_id>.npz

Each .npz contains:
  ecg     — filtered + z-scored ECG,        shape (5120,)   float32
  rpeaks  — R-peak indices within ecg,      shape (n_beats,) int32
  beats   — beat templates,                 shape (n_beats, 307) float32
  imu     — accelerometer + gyroscope,      shape (5120, 6) float32  (if available)

Two output files summarise the run:
  processed_index.csv   one row per SUCCESSFUL window (with sample_id, npz path,
                        n_beats, n_peaks), plus every column from index.csv
  preprocessing_log.csv one row per DROPPED window with the reason

Requires:  pip install pandas numpy scipy neurokit2 tqdm

Usage:
  python preprocess.py --data-root   /path/to/human-activity-Recognition \\
                       --dataset-dir /path/to/ecg_biometric_dataset

Options:
  --limit N          process only the first N rows of index.csv (smoke test)
  --only-batch N     process only rows where batch == N
  --skip-existing    skip windows whose output .npz already exists
  --no-imu           don't save IMU channels (halves disk usage)
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from tqdm import tqdm

# neurokit2 is chatty; silence its info logs
warnings.filterwarnings("ignore")
import neurokit2 as nk  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (Section 8 of the handoff)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 512
WINDOW_SIZE = 5120

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 4

NOTCH_FREQ_HZ = 50.0                # European power line (dataset from TU Dresden)
NOTCH_Q = 30.0

HP_CUTOFF_HZ = 0.67                 # Residual respiratory baseline wander

BEAT_BEFORE_SAMPLES = 102           # 200 ms at 512 Hz
BEAT_AFTER_SAMPLES = 205            # 400 ms at 512 Hz  (102 + 1 + 204 = 307)
BEAT_LENGTH = BEAT_BEFORE_SAMPLES + BEAT_AFTER_SAMPLES   # 307

MIN_PEAKS = 5                       # Discard windows with fewer R-peaks
MIN_PEAK_DIST_SAMPLES = 128         # 250 ms — physiological minimum RR interval

QC_MAX_NAN_FRAC = 0.10              # Drop if > 10% NaN
QC_STATUS_NORMAL = 128              # batch2 "normal" flag
QC_STATUS_MIN_NORMAL_FRAC = 0.80    # Drop if < 80% of samples flagged normal

# Column names (batch2 spec; batch4+ variants handled by presence check)
ECG_PRIMARY_COL = "ECG_LL-LA_24BIT_CAL"
STATUS_COL = "ECG_EMG_Status1_CAL"
IMU_COLS = [
    "Accel_WR_X_CAL", "Accel_WR_Y_CAL", "Accel_WR_Z_CAL",
    "Gyro_X_CAL",     "Gyro_Y_CAL",     "Gyro_Z_CAL",
]


# ---------------------------------------------------------------------------
# CSV loader — handles the four Shimmer export variants in this dataset
# ---------------------------------------------------------------------------
#
# The variants observed:
#   batch 2:   comma-sep, one header row, no units row, bare column names
#   batch 4:   comma-sep, one header row, one units row, bare column names
#   batch 7:   tab-sep,   one header row, no units row, Shimmer_XXXX_ prefixed
#   batches 1/5/6: Excel-exported — first line '"sep=\t"' declaration,
#                  then tab-sep with Shimmer_XXXX_ prefixed column names,
#                  units row possible on line 3
#
# The loader normalises all four to a DataFrame with:
#   - bare column names (Shimmer_XXXX_ prefix stripped)
#   - one row per real data sample (units row stripped if present)
#   - numeric dtype where possible

_SEP_DECL_RE = re.compile(rb'^"sep=')
_SHIMMER_PREFIX_RE = re.compile(r"^Shimmer_[A-Za-z0-9]+_(.+)$")

# Values that indicate a Shimmer "units" row rather than data.
_UNIT_TOKENS = {
    "ms", "s", "Hz", "V", "mV", "kPa",
    "local_flux", "Degrees Celsius", "deg/s", "g", "no_units",
}
_UNIT_PATTERN_RE = re.compile(r"^m/\(s\^?\d?\)")  # 'm/(s^2)', 'm/(s2)', etc.


def _looks_like_units_row(row_values) -> bool:
    """True if any cell in this row looks like a unit string, not a number."""
    for v in row_values:
        s = str(v).strip()
        if s in _UNIT_TOKENS or _UNIT_PATTERN_RE.match(s):
            return True
    return False


def load_source_csv(path: Path) -> pd.DataFrame:
    """Read a Shimmer CSV and normalise it.

    Returns a DataFrame whose row 0 is the first real data sample and whose
    columns are bare names like 'ECG_LL-LA_24BIT_CAL' (no Shimmer_XXXX_
    prefix). All numeric-looking columns are coerced to float; unparseable
    cells become NaN and are handled downstream by quality_check.
    """
    with open(path, "rb") as f:
        first_bytes = f.readline()

    if _SEP_DECL_RE.match(first_bytes):
        sep = "\t"
        skiprows: list[int] | None = [0]  # skip the '"sep=\t"' declaration
    else:
        line = first_bytes.decode("latin-1", errors="replace")
        # Whichever delimiter appears more often on the header line wins.
        sep = "\t" if line.count("\t") > line.count(",") else ","
        skiprows = None

    df = pd.read_csv(path, sep=sep, skiprows=skiprows,
                     encoding="latin-1", low_memory=False)

    # Strip Shimmer_XXXX_ prefix from column names.
    df.columns = [
        _SHIMMER_PREFIX_RE.sub(r"\1", c) if isinstance(c, str) else c
        for c in df.columns
    ]

    # Drop a leading units row if present.
    if len(df) > 0 and _looks_like_units_row(df.iloc[0].values):
        df = df.iloc[1:].reset_index(drop=True)

    # Coerce any object columns to numeric (bad values → NaN).
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Filter design (done once, reused for every window)
# ---------------------------------------------------------------------------

def design_filters(fs: float):
    bp = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
                btype="band", fs=fs, output="ba")
    notch = iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, fs=fs)
    hp = butter(BANDPASS_ORDER, HP_CUTOFF_HZ, btype="high", fs=fs, output="ba")
    return bp, notch, hp


def filter_ecg(signal: np.ndarray, filters) -> np.ndarray:
    """Bandpass → notch → high-pass. All zero-phase (filtfilt)."""
    bp, notch, hp = filters
    x = filtfilt(bp[0], bp[1], signal)
    x = filtfilt(notch[0], notch[1], x)
    x = filtfilt(hp[0], hp[1], x)
    return x


def zscore(signal: np.ndarray) -> np.ndarray:
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-9:
        return signal - mean
    return (signal - mean) / std


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def quality_check(ecg: np.ndarray, status: np.ndarray | None) -> tuple[bool, str]:
    """Return (is_bad, reason). Reason is empty when window is OK."""
    if len(ecg) < WINDOW_SIZE:
        return True, f"short_window({len(ecg)})"

    nan_frac = float(np.isnan(ecg).sum()) / len(ecg)
    if nan_frac > QC_MAX_NAN_FRAC:
        return True, f"nan_frac={nan_frac:.3f}"

    # Total-window flatline — a stricter rolling check would be nicer but
    # in practice electrode dropout kills the whole window's variance.
    if np.nanstd(ecg) < 1e-6:
        return True, "flatline"

    if status is not None and len(status) > 0:
        normal_frac = float(np.sum(status == QC_STATUS_NORMAL)) / len(status)
        if normal_frac < QC_STATUS_MIN_NORMAL_FRAC:
            return True, f"status_normal_frac={normal_frac:.3f}"

    return False, ""


def interpolate_small_nans(signal: np.ndarray) -> np.ndarray:
    """Linearly interpolate ≤ 10% NaNs. Called only after quality_check passes."""
    if not np.isnan(signal).any():
        return signal
    return (pd.Series(signal)
              .interpolate(method="linear", limit_direction="both")
              .to_numpy())


# ---------------------------------------------------------------------------
# R-peak detection + beat segmentation
# ---------------------------------------------------------------------------

def detect_rpeaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """neurokit2 R-peaks with a physiological minimum-distance filter applied
    afterwards to squash the occasional double-detection."""
    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs, correct_artifacts=True)
    except Exception:
        return np.array([], dtype=np.int32)

    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int32)
    if len(peaks) <= 1:
        return peaks

    kept = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] >= MIN_PEAK_DIST_SAMPLES:
            kept.append(p)
    return np.asarray(kept, dtype=np.int32)


def segment_beats(signal: np.ndarray, peaks: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (beats, kept_peaks). Beats crossing window edges are dropped."""
    beats = []
    kept = []
    for p in peaks:
        start = p - BEAT_BEFORE_SAMPLES
        end = p + BEAT_AFTER_SAMPLES
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])
        kept.append(p)
    if not beats:
        return (np.zeros((0, BEAT_LENGTH), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))
    return (np.asarray(beats, dtype=np.float32),
            np.asarray(kept, dtype=np.int32))


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process(data_root: Path, dataset_dir: Path, *,
            limit: int | None = None,
            only_batch: int | None = None,
            skip_existing: bool = False,
            save_imu: bool = True) -> None:
    index_path = dataset_dir / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"index.csv not found at {index_path}")

    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    proc_index_path = dataset_dir / "processed_index.csv"
    log_path = dataset_dir / "preprocessing_log.csv"

    # latin-1 never fails on any byte value — handles the em-dash in `notes`
    # that build_index.py wrote in Windows' cp1252 encoding.
    index = pd.read_csv(index_path, encoding="latin-1")
    # Stable sample_id = original position in index.csv (0-indexed).
    index = index.reset_index().rename(columns={"index": "sample_id"})

    if only_batch is not None:
        index = index[index["batch"] == only_batch].reset_index(drop=True)
        print(f"Filtered to batch {only_batch}: {len(index)} windows", file=sys.stderr)
    if limit is not None:
        index = index.head(limit).reset_index(drop=True)
        print(f"Limited to first {limit} windows", file=sys.stderr)

    filters = design_filters(SAMPLE_RATE)

    proc_rows: list[dict] = []
    log_rows: list[dict] = []
    drop_counts: Counter[str] = Counter()

    # Group by source file so each raw CSV is read only once.
    grouped = index.groupby("source_file", sort=False)
    file_progress = tqdm(grouped, desc="Files", unit="file")

    win_processed = 0
    win_dropped = 0

    for source_file, group in file_progress:
        full_path = data_root / source_file

        # Load the whole file once (handles all format variants).
        try:
            df = load_source_csv(full_path)
        except Exception as e:
            for _, row in group.iterrows():
                log_rows.append({**row.to_dict(),
                                 "reason": f"csv_read_error: {type(e).__name__}"})
                drop_counts["csv_read_error"] += 1
                win_dropped += 1
            continue

        if ECG_PRIMARY_COL not in df.columns:
            for _, row in group.iterrows():
                log_rows.append({**row.to_dict(),
                                 "reason": "missing_primary_ecg_column"})
                drop_counts["missing_primary_ecg_column"] += 1
                win_dropped += 1
            continue

        have_status = STATUS_COL in df.columns
        imu_present = [c for c in IMU_COLS if c in df.columns]

        for _, row in group.iterrows():
            sample_id = int(row["sample_id"])
            start = int(row["window_start_row"])
            end = int(row["window_end_row"])

            if end > len(df):
                log_rows.append({**row.to_dict(),
                                 "reason": f"row_range_beyond_file({len(df)})"})
                drop_counts["row_range_beyond_file"] += 1
                win_dropped += 1
                continue

            gid = row["global_subject_id"]
            out_dir = processed_dir / gid
            out_path = out_dir / f"{sample_id:06d}.npz"
            if skip_existing and out_path.exists():
                # Trust the file — record it in the processed index without re-running.
                proc_row = row.to_dict()
                proc_row["processed_path"] = str(out_path.relative_to(dataset_dir))
                proc_row["n_beats"] = -1        # unknown without loading
                proc_row["n_peaks"] = -1
                proc_row["reused"] = True
                proc_rows.append(proc_row)
                win_processed += 1
                continue

            ecg_raw = df[ECG_PRIMARY_COL].iloc[start:end].to_numpy(dtype=np.float64)
            status = (df[STATUS_COL].iloc[start:end].to_numpy()
                      if have_status else None)

            # Steps 2 (QC), 3–5 (filters), 6 (z-score)
            is_bad, reason = quality_check(ecg_raw, status)
            if is_bad:
                log_rows.append({**row.to_dict(), "reason": reason})
                drop_counts[reason.split("=")[0]] += 1
                win_dropped += 1
                continue

            ecg_clean_input = interpolate_small_nans(ecg_raw)
            try:
                ecg_filtered = filter_ecg(ecg_clean_input, filters)
            except Exception as e:
                log_rows.append({**row.to_dict(),
                                 "reason": f"filter_error: {type(e).__name__}"})
                drop_counts["filter_error"] += 1
                win_dropped += 1
                continue

            ecg_z = zscore(ecg_filtered)

            # Step 7: R-peak detection
            peaks = detect_rpeaks(ecg_z, SAMPLE_RATE)
            if len(peaks) < MIN_PEAKS:
                log_rows.append({**row.to_dict(),
                                 "reason": f"insufficient_peaks({len(peaks)})"})
                drop_counts["insufficient_peaks"] += 1
                win_dropped += 1
                continue

            # Step 8: beat segmentation
            beats, kept_peaks = segment_beats(ecg_z, peaks)
            if len(beats) < MIN_PEAKS:
                log_rows.append({**row.to_dict(),
                                 "reason": f"insufficient_beats_after_edge_trim({len(beats)})"})
                drop_counts["insufficient_beats_after_edge_trim"] += 1
                win_dropped += 1
                continue

            # Save
            out_dir.mkdir(exist_ok=True)
            payload = {
                "ecg": ecg_z.astype(np.float32),
                "rpeaks": kept_peaks.astype(np.int32),
                "beats": beats.astype(np.float32),
            }
            if save_imu and imu_present:
                imu = (df[imu_present].iloc[start:end]
                       .to_numpy(dtype=np.float32))
                payload["imu"] = imu
                payload["imu_channels"] = np.asarray(imu_present)
            np.savez_compressed(out_path, **payload)

            proc_row = row.to_dict()
            proc_row["processed_path"] = str(out_path.relative_to(dataset_dir))
            proc_row["n_beats"] = int(len(beats))
            proc_row["n_peaks"] = int(len(peaks))
            proc_row["reused"] = False
            proc_rows.append(proc_row)
            win_processed += 1

        file_progress.set_postfix(kept=win_processed, dropped=win_dropped)

    # ---- Write outputs ----
    if proc_rows:
        pd.DataFrame(proc_rows).to_csv(proc_index_path, index=False, encoding="utf-8")
    if log_rows:
        pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8")

    # ---- Summary ----
    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Processed: {win_processed} windows", file=sys.stderr)
    print(f"Dropped:   {win_dropped} windows", file=sys.stderr)
    if drop_counts:
        print("  drops by reason:", file=sys.stderr)
        for reason, count in drop_counts.most_common():
            print(f"    {count:>6}  {reason}", file=sys.stderr)
    if proc_rows:
        proc_df = pd.DataFrame(proc_rows)
        print(f"\nBy split: {dict(Counter(proc_df['split']))}", file=sys.stderr)
        print(f"By batch: {dict(Counter(proc_df['batch']))}", file=sys.stderr)
        n_subjects = proc_df["global_subject_id"].nunique()
        print(f"Unique subjects with ≥1 kept window: {n_subjects}", file=sys.stderr)
    print(f"\nWrote {proc_index_path.name} and {log_path.name}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess windows listed in index.csv → per-window .npz files."
    )
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Path to the cloned human-activity-Recognition/ repo")
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="Path to the dataset folder containing index.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N rows of index.csv (smoke test)")
    ap.add_argument("--only-batch", type=int, default=None,
                    choices=[1, 2, 4, 5, 6, 7],
                    help="Process only rows from this batch")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse existing .npz files (resume interrupted runs)")
    ap.add_argument("--no-imu", action="store_true",
                    help="Do not save IMU channels (halves disk usage)")
    args = ap.parse_args()

    process(args.data_root.resolve(),
            args.dataset_dir.resolve(),
            limit=args.limit,
            only_batch=args.only_batch,
            skip_existing=args.skip_existing,
            save_imu=not args.no_imu)


if __name__ == "__main__":
    main()
