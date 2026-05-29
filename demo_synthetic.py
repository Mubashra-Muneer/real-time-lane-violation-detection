"""
demo_synthetic.py
─────────────────
Generates a small set of synthetic road images (with lanes drawn on them),
runs the lane detector, and prints performance statistics.
Useful for testing/demonstrating the pipeline without a real video.
"""

import cv2
import numpy as np
import os
import time
from collections import deque
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lane_detector import (
    Config, get_roi_mask, process_frame, evaluate
)

# ── Synthetic image generator ───────────────────────────────────────────────

def make_road_frame(width=1280, height=720, offset_x=0, noise=True):
    """
    Draw a simple perspective road with two white lane lines.
    offset_x: positive → vehicle drifting right (simulates lane change).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky / road gradient
    sky_color  = (90, 60, 30)
    road_color = (60, 60, 60)
    for y in range(height):
        t = y / height
        c = tuple(int(sky_color[i] * (1 - t) + road_color[i] * t) for i in range(3))
        img[y, :] = c

    # Perspective vanishing-point lane lines
    vp_x = width // 2 + offset_x // 3
    vp_y = int(height * 0.42)

    left_bot  = (width // 2 - 380 + offset_x, height)
    right_bot = (width // 2 + 380 + offset_x, height)
    left_top  = (vp_x - 5,  vp_y)
    right_top = (vp_x + 5,  vp_y)

    # Dashed centre line
    steps = 16
    for i in range(steps):
        t1 = i       / steps
        t2 = (i + 0.45) / steps
        if i % 2 == 0:
            x1 = int(left_bot[0] + (vp_x - left_bot[0]) * t1)
            y1 = int(left_bot[1] + (vp_y - left_bot[1]) * t1)
            x2 = int(left_bot[0] + (vp_x - left_bot[0]) * t2)
            y2 = int(left_bot[1] + (vp_y - left_bot[1]) * t2)
            xr1 = int(right_bot[0] + (vp_x - right_bot[0]) * t1)
            yr1 = int(right_bot[1] + (vp_y - right_bot[1]) * t1)
            xr2 = int(right_bot[0] + (vp_x - right_bot[0]) * t2)
            yr2 = int(right_bot[1] + (vp_y - right_bot[1]) * t2)
            # Center dashes (yellow)
            cx1 = (x1 + xr1) // 2; cx2 = (x2 + xr2) // 2
            cy1 = (y1 + yr1) // 2; cy2 = (y2 + yr2) // 2
            cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 200, 200), 3)

    # Solid white lane lines
    cv2.line(img, left_bot,  left_top,  (220, 220, 220), 6)
    cv2.line(img, right_bot, right_top, (220, 220, 220), 6)

    # Road texture noise
    if noise:
        n = np.random.randint(-12, 12, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)

    return img


def generate_sequence(n_normal=40, n_change=20, change_magnitude=260):
    """
    Generate a list of (frame, is_violation) tuples.
    Simulates:  40 in-lane frames → 20 lane-change frames → 40 in-lane frames
    """
    seq = []
    # In-lane
    for _ in range(n_normal):
        seq.append((make_road_frame(offset_x=0), False))
    # Lane change (ramp offset)
    for i in range(n_change):
        off = int(change_magnitude * (i / n_change))
        seq.append((make_road_frame(offset_x=off), True))
    # Back in new lane
    for _ in range(n_normal):
        seq.append((make_road_frame(offset_x=change_magnitude), False))
    return seq


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic sequence …")
    sequence = generate_sequence(n_normal=40, n_change=20, change_magnitude=260)

    # Ground truth violation event (frames 40-59)
    gt_events = [(40, 59)]

    history_left  = deque(maxlen=Config.SMOOTH_FRAMES)
    history_right = deque(maxlen=Config.SMOOTH_FRAMES)

    os.makedirs("demo_output", exist_ok=True)
    os.makedirs("demo_frames", exist_ok=True)

    # Save individual frames as images
    for i, (frm, _) in enumerate(sequence):
        cv2.imwrite(f"demo_frames/frame_{i:04d}.jpg", frm)

    print(f"Saved {len(sequence)} demo frames to demo_frames/")

    # Run detector
    detected_violation_frames = []
    in_event = False
    consec   = 0
    events   = []
    min_consec = 6

    results  = []
    t_start  = time.time()

    for idx, (frame, gt_viol) in enumerate(sequence):
        roi_mask = get_roi_mask(frame.shape)
        annotated, edges_bgr, violation, offset = process_frame(
            frame, roi_mask, history_left, history_right
        )

        # Save combined result
        combined = np.hstack([
            cv2.resize(frame,     (640, 360)),
            cv2.resize(annotated, (640, 360)),
        ])
        cv2.imwrite(f"demo_output/result_{idx:04d}.jpg", combined)

        # Event tracking
        if violation:
            consec += 1
            if consec >= min_consec and not in_event:
                in_event = True
                event_start = idx - consec + 1
        else:
            if in_event:
                events.append((event_start, idx - 1))
                in_event = False
            consec = 0

        if violation:
            detected_violation_frames.append(idx)

        results.append({
            "frame": idx, "gt": gt_viol, "det": violation, "offset": offset
        })

    if in_event:
        events.append((event_start, len(sequence) - 1))

    elapsed = time.time() - t_start
    fps     = len(sequence) / elapsed

    # Frame-level accuracy
    correct = sum(1 for r in results if r["gt"] == r["det"])
    acc     = correct / len(results)

    tp_frames = sum(1 for r in results if r["gt"] and r["det"])
    fp_frames = sum(1 for r in results if not r["gt"] and r["det"])
    fn_frames = sum(1 for r in results if r["gt"] and not r["det"])
    tn_frames = sum(1 for r in results if not r["gt"] and not r["det"])

    print(f"\n{'='*55}")
    print("  DEMO RESULTS (Synthetic Sequence)")
    print(f"{'='*55}")
    print(f"  Frames processed : {len(sequence)}")
    print(f"  Processing time  : {elapsed:.2f} s")
    print(f"  Throughput       : {fps:.1f} fps")
    print(f"  Correct frames   : {correct}/{len(sequence)}  ({acc:.1%})")
    print(f"  TP frames        : {tp_frames}")
    print(f"  FP frames        : {fp_frames}  ← false violation frames")
    print(f"  FN frames        : {fn_frames}  ← missed violation frames")
    print(f"  TN frames        : {tn_frames}")
    print(f"{'─'*55}")
    print(f"  Lane-change events detected: {events}")
    print(f"  Ground-truth events        : {gt_events}")
    evaluate(events, gt_events, len(sequence))
    print(f"\n  Output images saved to: demo_output/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
