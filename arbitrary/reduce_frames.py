"""
reduce_frames.py
----------------
Takes an MP4 file, identifies the least essential frames using SSIM
(Structural Similarity Index), enforces a minimum spacing rule so no
two adjacent frames are dropped, and writes the retained frames as
numbered JPGs into an output folder. Also writes a text log of removed
frame numbers.

Usage:
    python reduce_frames.py <input.mp4> [options]

Options:
    --output-dir   DIR     Output folder (default: <input_stem>_frames/)
    --drop-pct     FLOAT   Percentage of frames to drop, 0–100 (default: 30)
    --min-spacing  INT     Min frames between any two dropped frames (default: 3)
    --jpg-quality  INT     JPEG save quality, 1–100 (default: 92)

Requirements:
    pip install opencv-python scikit-image
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce video frames by dropping least-essential ones."
    )
    parser.add_argument("input", help="Path to the input .mp4 file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder (default: <input_stem>_frames/)",
    )
    parser.add_argument(
        "--drop-pct",
        type=float,
        default=30.0,
        help="Percentage of frames to drop (default: 30)",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=3,
        help="Minimum number of frames between two dropped frames (default: 3)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=92,
        help="JPEG output quality 1–100 (default: 92)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------


def read_frames(video_path: str):
    """Read all frames from the video as grayscale numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    frames = []
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Reading {total} frames at {fps:.2f} fps …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  … read {frame_idx}/{total}")

    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames.")
    return frames, fps


# ---------------------------------------------------------------------------
# SSIM-based difference scoring
# ---------------------------------------------------------------------------


def compute_ssim_scores(frames):
    """
    Compute per-frame SSIM score against the previous frame.
    A HIGH SSIM score (close to 1.0) means the frame is very similar to
    its predecessor → it is a candidate for removal.

    Frame 0 gets score 0.0 (always kept — no predecessor).
    """
    n = len(frames)
    scores = np.zeros(n, dtype=np.float64)

    print(f"[INFO] Computing SSIM scores for {n - 1} frame pairs …")

    for i in range(1, n):
        score = ssim(frames[i - 1], frames[i], full=True)[0]

        scores[i] = score  # higher → more similar → less essential
        if i % 200 == 0:
            print(f"  … scored {i}/{n - 1}")

    print(
        f"[INFO] SSIM scoring complete. "
        f"Min={scores[1:].min():.4f}, Max={scores[1:].max():.4f}, "
        f"Mean={scores[1:].mean():.4f}"
    )
    return scores


# ---------------------------------------------------------------------------
# Frame selection with minimum-spacing enforcement
# ---------------------------------------------------------------------------


def select_frames_to_drop(scores, drop_pct: float, min_spacing: int):
    """
    Given per-frame SSIM scores, return the set of frame indices to DROP.

    Strategy:
      1. Compute how many frames we want to drop.
      2. Sort candidate frames by score descending (most similar first).
      3. Greedily add them to the drop set, skipping any candidate that
         would violate the minimum-spacing rule.
      4. Frame 0 and the last frame are never dropped.
    """
    n = len(scores)
    target_drops = int(round(n * drop_pct / 100.0))

    # Frames 0 and n-1 are never droppable
    candidates = list(range(1, n - 1))

    # Sort by SSIM score descending: most similar (least essential) first
    candidates.sort(key=lambda i: scores[i], reverse=True)

    dropped = set()

    def too_close(idx):
        """True if idx is within min_spacing of any already-dropped frame."""
        for d in dropped:
            if abs(idx - d) < min_spacing:
                return True
        return False

    for candidate in candidates:
        if len(dropped) >= target_drops:
            break
        if not too_close(candidate):
            dropped.add(candidate)

    print(
        f"[INFO] Target drops: {target_drops}  |  "
        f"Actual drops (after spacing): {len(dropped)}  |  "
        f"Retained: {n - len(dropped)}"
    )
    return dropped


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(frames, dropped: set, output_dir: Path, jpg_quality: int):
    """Save retained frames as JPGs and write the removed-frames log."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    kept = sorted(i for i in range(n) if i not in dropped)
    removed = sorted(dropped)

    print(f"[INFO] Saving {len(kept)} frames to {output_dir} …")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]

    for frame_num in kept:
        filename = output_dir / f"frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(filename), frames[frame_num], encode_params)

    # Write removed-frames log
    log_path = output_dir / "removed_frames.txt"
    with open(log_path, "w") as f:
        f.write(f"Total frames in source : {n}\n")
        f.write(f"Frames retained        : {len(kept)}\n")
        f.write(f"Frames removed         : {len(removed)}\n")
        f.write("\n--- Removed frame numbers (0-indexed) ---\n")
        for fn in removed:
            f.write(f"{fn}\n")

    print(f"[INFO] Saved removed-frames log to {log_path}")
    print(f"[DONE] {len(kept)} frames written, {len(removed)} removed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.exit(f"[ERROR] File not found: {input_path}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_frames"
    )

    if not (0 < args.drop_pct < 100):
        sys.exit("[ERROR] --drop-pct must be between 0 and 100 (exclusive).")
    if args.min_spacing < 1:
        sys.exit("[ERROR] --min-spacing must be at least 1.")
    if not (1 <= args.jpg_quality <= 100):
        sys.exit("[ERROR] --jpg-quality must be between 1 and 100.")

    frames, fps = read_frames(str(input_path))

    if len(frames) < 3:
        sys.exit("[ERROR] Video has fewer than 3 frames — nothing to reduce.")

    scores = compute_ssim_scores(frames)
    dropped = select_frames_to_drop(scores, args.drop_pct, args.min_spacing)
    save_outputs(frames, dropped, output_dir, args.jpg_quality)


if __name__ == "__main__":
    main()
