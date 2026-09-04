#!/usr/bin/env python3
"""build_index.py — ECG biometric dataset ingestion.

Walks a cloned human-activity-Recognition repository, parses filenames per
batch, assigns unified global subject IDs (S001–S083), standardizes activity
labels, computes 10-second / 50%-overlap window ranges, assigns train/val/test
split, and writes:

    <out-dir>/index.csv         one row per window
    <out-dir>/subject_map.csv   one row per global subject
    <out-dir>/ingestion_log.txt human-readable log of skips and anomalies

No signal data is copied into the index — each window row stores
(source_file, window_start_row, window_end_row) so the training pipeline
loads samples on-the-fly and the index stays a few MB.

Usage:
    python build_index.py --data-root /path/to/human-activity-Recognition \\
                          --out-dir   /path/to/ecg_biometric_dataset

Assumes each source CSV has exactly one header row. Adjust HEADER_ROWS if not.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Windowing constants (Section 6 of the handoff)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 512                        # Hz
WINDOW_SECONDS = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS   # 5120 samples
STEP_SIZE = WINDOW_SIZE // 2                 # 2560 samples (50% overlap)
HEADER_ROWS = 1                              # Shimmer CSVs: single header row


# ---------------------------------------------------------------------------
# Subject-ID maps (Section 4 of the handoff)
# ---------------------------------------------------------------------------

# Batch 2 first names, in the handoff's canonical order → S009–S024.
# These same names are reused in batch 4 for cross-session recordings — they
# keep their batch-2 global ID (that's the whole point of them).
BATCH2_NAMES = [
    "Antonia", "Daniel", "Elisabeth", "Freya", "Gina", "Horst",
    "Johanna", "Julia", "Kay", "Marie", "Nhung", "Paul",
    "Paulus", "Peggy", "Sebastian", "Sophie",
]
NAME_TO_GLOBAL: dict[str, str] = {
    name.lower(): f"S{i + 9:03d}" for i, name in enumerate(BATCH2_NAMES)
}

# Subjects confirmed by the handoff to appear in both batch 2 and batch 4.
# (Used for a sanity flag, not for logic — cross-session is derived from data.)
CROSS_SESSION_NAMES = {
    "antonia", "daniel", "elisabeth", "johanna", "julia", "kay",
    "marie", "nhung", "paul", "sebastian", "sophie",
}

# Batch 4 new subjects: original S35–S54 → S025–S044.
BATCH4_SCODE_TO_GLOBAL: dict[str, str] = {
    f"S{35 + i}": f"S{25 + i:03d}" for i in range(20)
}

# Batch 5 first names → S045–S052. Handoff lists them in this order.
BATCH5_NAMES = [
    "chyntia", "henri", "igor", "joseph",
    "michelle", "rachel", "samuel", "zebulun",
]
BATCH5_NAME_TO_GLOBAL: dict[str, str] = {
    name: f"S{i + 45:03d}" for i, name in enumerate(BATCH5_NAMES)
}

# Batch 6 anonymized codes s1–s15 → S053–S067.
BATCH6_S_TO_GLOBAL: dict[str, str] = {
    f"s{i}": f"S{52 + i:03d}" for i in range(1, 16)
}

# Batch 7 anonymized codes s00–s15 → S068–S083.
BATCH7_S_TO_GLOBAL: dict[str, str] = {
    f"s{i:02d}": f"S{68 + i:03d}" for i in range(16)
}


# ---------------------------------------------------------------------------
# Activity label standardization (Section 6 of the handoff)
# ---------------------------------------------------------------------------

ACTIVITY_MAP: dict[str, str] = {
    # English (case sensitivity handled below with .lower() fallback)
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "running": "running",
    "skipping": "skipping",
    "bending": "bending",
    "badminton": "badminton",
    "stairs_up": "stairs_up",
    "stairs up": "stairs_up",
    "stairs_down": "stairs_down",
    "stairs down": "stairs_down",
    "stairs": "stairs_combined",     # batch 1 only — up + down mixed
    # German (batch 1 filenames)
    "sitzen": "sitting",
    "laufen": "running",
    "gehen": "walking",
    "aufstehen": "standing",
    "beugen": "bending",
    "treppe": "stairs_combined",
    "springen": "skipping",
    # Batch 4 activity codes
    "SI": "sitting", "ST": "standing", "W": "walking", "R": "running",
    "CSU": "stairs_up", "CSD": "stairs_down", "SK": "skipping", "BAD": "badminton",
}

# Batch 7 folders that contain a continuous recording of several activities.
MULTI_ACTIVITY_FOLDERS = {
    "WalkingRunning",
    "WalkingSkippingRunning",
    "WalkingClimbUPClimbDOWN",
    "WalkingClimbDOWNSkipping",
    "WalkingStandingClimbUP",
}


def normalize_activity(raw: str) -> Optional[str]:
    """Map a raw folder or filename token to the canonical activity label."""
    key = raw.strip()
    if key in ACTIVITY_MAP:
        return ACTIVITY_MAP[key]
    if key.lower() in ACTIVITY_MAP:
        return ACTIVITY_MAP[key.lower()]
    return None


# ---------------------------------------------------------------------------
# Recording record — one source CSV, before windowing
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    global_subject_id: str
    original_id: str
    batch: int
    activity: str
    is_multi_activity: bool
    session: int
    source_file: str    # path relative to --data-root
    split: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Per-batch filename parsers
# ---------------------------------------------------------------------------

_BATCH1_RE = re.compile(r"^(\d{2})_([A-Za-zÄÖÜäöüß]+)_[0-9A-Za-z]+_center$")
_BATCH4_SCODE_RE = re.compile(r"^(S\d{2})_([A-Z]+)$")
_BATCH6_RE = re.compile(r"^([A-Za-z_ ]+?)_(s\d+)$")
_BATCH7_RE = re.compile(r"^(s\d{2})$")


def _match_name_suffix(stem: str, names: Iterable[str], sep: str
                       ) -> Optional[tuple[str, str]]:
    """If stem ends with `{sep}{name}` for some name in names (case-insensitive),
    return (prefix, matched_name_lower). Longest match wins so 'Paulus' beats
    'Paul' on 'sitting_Paulus'."""
    stem_lower = stem.lower()
    best: Optional[tuple[str, str]] = None
    for name in names:
        suffix = f"{sep}{name.lower()}"
        if stem_lower.endswith(suffix):
            prefix = stem[: len(stem) - len(suffix)]
            if best is None or len(name) > len(best[1]):
                best = (prefix, name.lower())
    return best


def parse_batch1(path: Path, activity_folder: str) -> Optional[Recording]:
    """batch1: {NN}_{german_activity}_{sensor}_center.csv"""
    m = _BATCH1_RE.match(path.stem)
    if not m:
        return None
    num, german_activity = m.group(1), m.group(2)
    activity = normalize_activity(german_activity) or normalize_activity(activity_folder)
    if activity is None:
        return None
    return Recording(
        global_subject_id=f"S{int(num):03d}",
        original_id=num,
        batch=1,
        activity=activity,
        is_multi_activity=False,
        session=1,
        source_file="",  # filled in by caller
        split="train",
    )


def parse_batch2(path: Path, activity_folder: str) -> Optional[Recording]:
    """batch2: {activity}_{FirstName}.csv (activity may contain underscores)."""
    match = _match_name_suffix(path.stem, BATCH2_NAMES, sep="_")
    if match is None:
        return None
    activity_token, name_lower = match
    activity = normalize_activity(activity_token) or normalize_activity(activity_folder)
    if activity is None:
        return None
    return Recording(
        global_subject_id=NAME_TO_GLOBAL[name_lower],
        original_id=name_lower.capitalize(),
        batch=2,
        activity=activity,
        is_multi_activity=False,
        session=1,
        source_file="",
        split="train",
    )


def parse_batch4(path: Path, activity_folder: str) -> Optional[Recording]:
    """batch4 has two filename formats:
       (a) S{NN}_{code}.csv           — new subjects,        session 1
       (b) {activity}_{FirstName}.csv — cross-session redo,  session 2
    """
    stem = path.stem
    # Format (a): S-code with activity abbreviation
    m = _BATCH4_SCODE_RE.match(stem)
    if m:
        scode, activity_code = m.group(1), m.group(2)
        gid = BATCH4_SCODE_TO_GLOBAL.get(scode)
        if gid is None:
            return None
        activity = normalize_activity(activity_code) or normalize_activity(activity_folder)
        if activity is None:
            return None
        return Recording(
            global_subject_id=gid,
            original_id=scode,
            batch=4,
            activity=activity,
            is_multi_activity=False,
            session=1,
            source_file="",
            split="val",
        )
    # Format (b): first-name cross-session file
    match = _match_name_suffix(stem, BATCH2_NAMES, sep="_")
    if match is None:
        return None
    activity_token, name_lower = match
    activity = normalize_activity(activity_token) or normalize_activity(activity_folder)
    if activity is None:
        return None
    # Flag known mislabels: filename activity disagrees with containing folder.
    notes = ""
    folder_activity = normalize_activity(activity_folder)
    if folder_activity is not None and folder_activity != activity:
        notes = (f"filename says {activity!r} but folder is {folder_activity!r} — "
                 f"inspect IMU to confirm true activity")
    return Recording(
        global_subject_id=NAME_TO_GLOBAL[name_lower],
        original_id=name_lower.capitalize(),
        batch=4,
        activity=activity,
        is_multi_activity=False,
        session=2,        # cross-session re-recording
        source_file="",
        split="test",
        notes=notes,
    )


def parse_batch5(path: Path, activity_folder: str) -> Optional[Recording]:
    """batch5: {activity}-{firstname}.csv (hyphen separator, not underscore)."""
    match = _match_name_suffix(path.stem, BATCH5_NAMES, sep="-")
    if match is None:
        return None
    activity_token, name_lower = match
    activity = normalize_activity(activity_token) or normalize_activity(activity_folder)
    if activity is None:
        return None
    return Recording(
        global_subject_id=BATCH5_NAME_TO_GLOBAL[name_lower],
        original_id=name_lower,
        batch=5,
        activity=activity,
        is_multi_activity=False,
        session=1,
        source_file="",
        split="train",
    )


def parse_batch6(path: Path, activity_folder: str) -> Optional[Recording]:
    """batch6: {activity}_{sN}.csv, where activity may contain a space."""
    m = _BATCH6_RE.match(path.stem)
    if not m:
        return None
    activity_token, s_code = m.group(1), m.group(2)
    gid = BATCH6_S_TO_GLOBAL.get(s_code)
    if gid is None:
        return None
    activity = normalize_activity(activity_token) or normalize_activity(activity_folder)
    if activity is None:
        return None
    return Recording(
        global_subject_id=gid,
        original_id=s_code,
        batch=6,
        activity=activity,
        is_multi_activity=False,
        session=1,
        source_file="",
        split="train",
    )


def parse_batch7(path: Path, activity_folder: str) -> Optional[Recording]:
    """batch7: {sNN}.csv, folder name = activity (may be multi-activity)."""
    m = _BATCH7_RE.match(path.stem)
    if not m:
        return None
    s_code = m.group(1)
    gid = BATCH7_S_TO_GLOBAL.get(s_code)
    if gid is None:
        return None
    is_multi = activity_folder in MULTI_ACTIVITY_FOLDERS
    if is_multi:
        activity = "multi_activity"
        notes = f"multi-activity folder: {activity_folder}"
    else:
        activity_resolved = normalize_activity(activity_folder)
        if activity_resolved is None:
            return None
        activity = activity_resolved
        notes = ""
    return Recording(
        global_subject_id=gid,
        original_id=s_code,
        batch=7,
        activity=activity,
        is_multi_activity=is_multi,
        session=1,
        source_file="",
        split="test",
        notes=notes,
    )


BATCH_PARSERS = {
    1: parse_batch1,
    2: parse_batch2,
    4: parse_batch4,
    5: parse_batch5,
    6: parse_batch6,
    7: parse_batch7,
}


# ---------------------------------------------------------------------------
# Filesystem walking
# ---------------------------------------------------------------------------

def find_batch_dirs(data_root: Path) -> dict[int, Path]:
    """Return {batch_number: path} for every batch{N} folder present. Batch 3
    is expected to be absent (handoff Section 2)."""
    found: dict[int, Path] = {}
    for n in (1, 2, 4, 5, 6, 7):
        p = data_root / f"batch{n}"
        if p.is_dir():
            found[n] = p
    return found


def iter_activity_dirs(batch_dir: Path) -> Iterable[Path]:
    for entry in sorted(batch_dir.iterdir()):
        if entry.is_dir():
            yield entry


def iter_csvs(activity_dir: Path) -> Iterable[Path]:
    for entry in sorted(activity_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".csv":
            yield entry


# ---------------------------------------------------------------------------
# Row counting & window computation
# ---------------------------------------------------------------------------

def count_data_rows(csv_path: Path) -> int:
    """Fast raw line count minus header row(s). Uses buffered binary read so
    it's ~10x quicker than pandas for the ~60k-row Shimmer CSVs."""
    with open(csv_path, "rb") as f:
        total = sum(1 for _ in f)
    return max(total - HEADER_ROWS, 0)


def compute_windows(n_rows: int) -> list[tuple[int, int]]:
    """Return [(start, end), ...] for non-truncated 10 s / 50%-overlap windows.
    Rows are indexed relative to the *data* portion (header excluded)."""
    if n_rows < WINDOW_SIZE:
        return []
    out = []
    start = 0
    while start + WINDOW_SIZE <= n_rows:
        out.append((start, start + WINDOW_SIZE))
        start += STEP_SIZE
    return out


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------

def build(data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    subject_map_path = out_dir / "subject_map.csv"
    log_path = out_dir / "ingestion_log.txt"

    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, file=sys.stderr)

    batch_dirs = find_batch_dirs(data_root)
    if not batch_dirs:
        raise SystemExit(f"No batch{{1..7}} folders found under {data_root}")
    log(f"Data root: {data_root}")
    log(f"Found batches: {sorted(batch_dirs)}")

    # ---- Parse every CSV filename ----
    recordings: list[Recording] = []
    skipped: list[tuple[Path, str]] = []

    for batch_num, batch_dir in batch_dirs.items():
        parser = BATCH_PARSERS[batch_num]
        n_before = len(recordings)
        for activity_dir in iter_activity_dirs(batch_dir):
            for csv_path in iter_csvs(activity_dir):
                rec = parser(csv_path, activity_dir.name)
                if rec is None:
                    skipped.append((csv_path, "unparsable filename"))
                    continue
                rec.source_file = str(csv_path.relative_to(data_root))
                recordings.append(rec)
        log(f"  batch{batch_num}: {len(recordings) - n_before} recordings parsed")

    log(f"Total parsed: {len(recordings)}   skipped: {len(skipped)}")
    for p, reason in skipped:
        log(f"  SKIP {p.relative_to(data_root)}: {reason}")

    # ---- Write index.csv (one row per window) ----
    windows_by_split: Counter[str] = Counter()
    windows_by_batch: Counter[int] = Counter()
    short_files: list[str] = []

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_subject_id", "original_id", "batch", "activity",
            "is_multi_activity", "session", "source_file",
            "window_id", "window_start_row", "window_end_row",
            "split", "notes",
        ])
        for rec in recordings:
            n_rows = count_data_rows(data_root / rec.source_file)
            wins = compute_windows(n_rows)
            if not wins:
                short_files.append(f"{rec.source_file} ({n_rows} rows)")
                continue
            for wid, (s, e) in enumerate(wins):
                writer.writerow([
                    rec.global_subject_id, rec.original_id, rec.batch, rec.activity,
                    rec.is_multi_activity, rec.session, rec.source_file,
                    wid, s, e, rec.split, rec.notes,
                ])
                windows_by_split[rec.split] += 1
                windows_by_batch[rec.batch] += 1

    total_windows = sum(windows_by_split.values())
    log(f"Wrote {total_windows} windows to {index_path.name}")
    log(f"  by split: {dict(windows_by_split)}")
    log(f"  by batch: {dict(windows_by_batch)}")
    if short_files:
        log(f"  {len(short_files)} file(s) too short for one window:")
        for s in short_files:
            log(f"    SHORT {s}")

    # ---- Aggregate subject_map.csv ----
    per_subject: dict[str, dict] = defaultdict(
        lambda: {"batches": set(), "original_ids": set(),
                 "sessions": set(), "activities": set(), "notes": []}
    )
    for rec in recordings:
        entry = per_subject[rec.global_subject_id]
        entry["batches"].add(rec.batch)
        entry["original_ids"].add(rec.original_id)
        entry["sessions"].add(rec.session)
        entry["activities"].add(rec.activity)
        if rec.notes:
            entry["notes"].append(rec.notes)

    with open(subject_map_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_subject_id", "batches", "original_ids",
            "is_cross_session", "first_appearance_batch",
            "n_sessions", "activities", "notes",
        ])
        for gid in sorted(per_subject):
            e = per_subject[gid]
            batches_sorted = sorted(e["batches"])
            # Cross-session = same subject recorded in ≥2 batches OR ≥2 sessions.
            is_cross = len(e["batches"]) > 1 or len(e["sessions"]) > 1
            unique_notes = sorted({n for n in e["notes"]})
            writer.writerow([
                gid,
                "|".join(str(b) for b in batches_sorted),
                "|".join(sorted(e["original_ids"])),
                is_cross,
                batches_sorted[0],
                len(e["sessions"]),
                "|".join(sorted(e["activities"])),
                " ; ".join(unique_notes),
            ])
    log(f"Wrote {len(per_subject)} subjects to {subject_map_path.name}")

    # Sanity: any expected cross-session subject NOT flagged?
    for name in CROSS_SESSION_NAMES:
        gid = NAME_TO_GLOBAL[name]
        if gid in per_subject and len(per_subject[gid]["batches"]) < 2:
            log(f"  NOTE: {gid} ({name.capitalize()}) expected cross-session "
                f"but only found in batch(es) {sorted(per_subject[gid]['batches'])}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"Log written to {log_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build ECG biometric identification index from the "
                    "human-activity-Recognition dataset."
    )
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Path to the cloned human-activity-Recognition/ repo")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write index.csv, subject_map.csv, log")
    args = ap.parse_args()
    build(args.data_root.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
