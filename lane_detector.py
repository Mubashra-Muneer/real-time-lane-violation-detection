"""
Road Lane Violation Detection using Canny Edge Detection and Hough Transform
Assignment Part B - Tasks B1, B3, B4
"""

import cv2
import numpy as np
import os
import time
import argparse
from collections import deque


# ──────────────────────────────────────────────
# Configuration / Tunable Parameters
# ──────────────────────────────────────────────
class Config:
    # Canny edge detection
    CANNY_LOW  = 50
    CANNY_HIGH = 150

    # Gaussian blur kernel size (must be odd)
    BLUR_KERNEL = (5, 5)

    # ROI – fraction of image height where ROI starts (top of trapezoid)
    ROI_TOP_RATIO    = 0.58   # row = height * ROI_TOP_RATIO
    ROI_BOTTOM_RATIO = 1.00   # row = height * ROI_BOTTOM_RATIO (bottom of frame)
    ROI_TOP_WIDTH    = 0.10   # half-width at top, fraction of image width
    ROI_BOTTOM_WIDTH = 0.48   # half-width at bottom, fraction of image width

    # Hough Transform
    HOUGH_RHO         = 1
    HOUGH_THETA       = np.pi / 180
    HOUGH_THRESHOLD   = 20
    HOUGH_MIN_LENGTH  = 30
    HOUGH_MAX_GAP     = 100

    # Lane line slope filtering
    SLOPE_MIN = 0.3   # ignore near-horizontal segments
    SLOPE_MAX = 2.5   # ignore near-vertical segments

    # Lane violation tolerance (pixels from lane centre)
    VIOLATION_TOLERANCE = 60

    # Smoothing: number of past frames for line averaging
    SMOOTH_FRAMES = 8


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def get_roi_mask(frame_shape):
    """Return a binary mask for the trapezoidal ROI."""
    h, w = frame_shape[:2]
    top_y    = int(h * Config.ROI_TOP_RATIO)
    bot_y    = int(h * Config.ROI_BOTTOM_RATIO)
    top_half = int(w * Config.ROI_TOP_WIDTH)
    bot_half = int(w * Config.ROI_BOTTOM_WIDTH)
    cx = w // 2

    pts = np.array([[
        (cx - bot_half, bot_y),
        (cx + bot_half, bot_y),
        (cx + top_half, top_y),
        (cx - top_half, top_y),
    ]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def apply_canny(frame):
    """Convert to grayscale, blur, and apply Canny edge detection."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, Config.BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, Config.CANNY_LOW, Config.CANNY_HIGH)
    return edges


def detect_lines(edges, mask):
    """Apply Hough Transform inside the ROI and return raw line segments."""
    roi_edges = cv2.bitwise_and(edges, edges, mask=mask)
    lines = cv2.HoughLinesP(
        roi_edges,
        Config.HOUGH_RHO,
        Config.HOUGH_THETA,
        Config.HOUGH_THRESHOLD,
        minLineLength=Config.HOUGH_MIN_LENGTH,
        maxLineGap=Config.HOUGH_MAX_GAP,
    )
    return lines, roi_edges


def slope_intercept(x1, y1, x2, y2):
    """Return (slope, intercept) of a line segment."""
    if x2 == x1:
        return None, None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def separate_and_average_lines(lines, h):
    """
    Separate Hough segments into left / right lanes.
    Average into a single (slope, intercept) per side.
    Returns: (left_line, right_line) each = (x1,y1,x2,y2) or None
    """
    left_params, right_params = [], []

    if lines is None:
        return None, None

    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        slope, intercept = slope_intercept(x1, y1, x2, y2)
        if slope is None:
            continue
        if abs(slope) < Config.SLOPE_MIN or abs(slope) > Config.SLOPE_MAX:
            continue
        length = np.hypot(x2 - x1, y2 - y1)
        if slope < 0:                     # negative slope → left lane
            left_params.append((slope, intercept, length))
        else:                             # positive slope → right lane
            right_params.append((slope, intercept, length))

    def weighted_avg(params):
        if not params:
            return None
        slopes, intercepts, lengths = zip(*params)
        total = sum(lengths)
        avg_slope     = sum(s * l for s, l in zip(slopes, lengths))     / total
        avg_intercept = sum(i * l for i, l in zip(intercepts, lengths)) / total
        return avg_slope, avg_intercept

    def to_coords(si, h):
        if si is None:
            return None
        slope, intercept = si
        y1 = h
        y2 = int(h * Config.ROI_TOP_RATIO)
        if slope == 0:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)

    left_line  = to_coords(weighted_avg(left_params),  h)
    right_line = to_coords(weighted_avg(right_params), h)
    return left_line, right_line


def extrapolate_smoothed(history_left, history_right, new_left, new_right):
    """Push new detections into deques and return smoothed lines."""
    if new_left  is not None: history_left.append(new_left)
    if new_right is not None: history_right.append(new_right)

    def avg(hist):
        if not hist:
            return None
        arr = np.array(hist, dtype=float)
        return tuple(np.mean(arr, axis=0).astype(int))

    return avg(history_left), avg(history_right)


def detect_violation(left_line, right_line, frame_width):
    """
    Estimate lane centre from bottom endpoints of left/right lines.
    Vehicle is assumed at frame centre (camera-mounted).
    Returns: (violation: bool, offset_px: int, lane_centre_x: int)
    """
    vehicle_x = frame_width // 2

    if left_line is None or right_line is None:
        return False, 0, vehicle_x  # cannot determine

    left_bottom_x  = left_line[0]
    right_bottom_x = right_line[0]
    lane_centre_x  = (left_bottom_x + right_bottom_x) // 2
    offset         = vehicle_x - lane_centre_x        # + → drifting right, - → left
    violation      = abs(offset) > Config.VIOLATION_TOLERANCE

    return violation, offset, lane_centre_x


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def draw_overlay(frame, left_line, right_line, violation, offset, lane_centre_x, roi_mask):
    """Draw lane lines, filled polygon, ROI outline, and violation status."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw ROI outline (thin blue)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 100, 0), 1)

    # Fill lane polygon (green / red)
    if left_line is not None and right_line is not None:
        poly_color = (0, 80, 255) if violation else (0, 180, 0)
        pts = np.array([[
            (left_line[0],  left_line[1]),
            (right_line[0], right_line[1]),
            (right_line[2], right_line[3]),
            (left_line[2],  left_line[3]),
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, pts, poly_color)

    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Draw lane lines
    if left_line is not None:
        cv2.line(frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]),
                 (0, 255, 255), 3)
    if right_line is not None:
        cv2.line(frame, (right_line[0], right_line[1]), (right_line[2], right_line[3]),
                 (0, 255, 255), 3)

    # Draw vehicle centre and lane centre lines
    cv2.line(frame, (w // 2, h - 20), (w // 2, h - 60), (255, 255, 0), 2)
    cv2.line(frame, (lane_centre_x, h - 20), (lane_centre_x, h - 60), (0, 165, 255), 2)

    # Status text
    status_txt   = "VIOLATION!" if violation else "IN LANE"
    status_color = (0, 0, 255)  if violation else (0, 255, 0)
    cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Status : {status_txt}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
    cv2.putText(frame, f"Offset : {offset:+d} px", (15, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    return frame


# ──────────────────────────────────────────────
# Core processing function (single frame)
# ──────────────────────────────────────────────

def process_frame(frame, roi_mask, history_left, history_right):
    """
    Full pipeline for one frame.
    Returns: annotated_frame, edges_frame, violation (bool), offset (int)
    """
    h, w = frame.shape[:2]

    edges                  = apply_canny(frame)
    lines, roi_edges       = detect_lines(edges, roi_mask)
    raw_left, raw_right    = separate_and_average_lines(lines, h)
    left_line, right_line  = extrapolate_smoothed(
        history_left, history_right, raw_left, raw_right
    )
    violation, offset, lane_cx = detect_violation(left_line, right_line, w)

    annotated = draw_overlay(
        frame.copy(), left_line, right_line, violation, offset, lane_cx, roi_mask
    )

    # Build edges display (3-ch for writing)
    edges_bgr = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)

    return annotated, edges_bgr, violation, offset


# ──────────────────────────────────────────────
# Image folder processing (Task B1 / B3)
# ──────────────────────────────────────────────

def process_image_folder(folder_path, output_folder="output_images", show=True):
    """Process all images in a folder and save results."""
    os.makedirs(output_folder, exist_ok=True)
    exts   = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    images = sorted([f for f in os.listdir(folder_path)
                     if f.lower().endswith(exts)])

    if not images:
        print(f"No images found in {folder_path}")
        return

    print(f"Found {len(images)} images in '{folder_path}'")

    history_left  = deque(maxlen=Config.SMOOTH_FRAMES)
    history_right = deque(maxlen=Config.SMOOTH_FRAMES)

    total_violations = 0
    t_start = time.time()

    for idx, fname in enumerate(images):
        fpath = os.path.join(folder_path, fname)
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"  [skip] Cannot read {fname}")
            continue

        roi_mask = get_roi_mask(frame.shape)
        annotated, edges_bgr, violation, offset = process_frame(
            frame, roi_mask, history_left, history_right
        )

        if violation:
            total_violations += 1

        # Add frame number
        cv2.putText(annotated, f"Frame {idx + 1}/{len(images)}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Side-by-side: original | annotated | edges
        combined = np.hstack([
            cv2.resize(frame,      (640, 360)),
            cv2.resize(annotated,  (640, 360)),
            cv2.resize(edges_bgr,  (640, 360)),
        ])
        out_path = os.path.join(output_folder, f"result_{idx:05d}_{fname}")
        cv2.imwrite(out_path, combined)

        if show:
            cv2.imshow("Lane Detection (L: original | C: annotated | R: edges)", combined)
            key = cv2.waitKey(80)
            if key == ord('q'):
                break

        if (idx + 1) % 10 == 0:
            elapsed   = time.time() - t_start
            fps       = (idx + 1) / elapsed
            print(f"  Processed {idx + 1}/{len(images)} | {fps:.1f} fps | "
                  f"violations so far: {total_violations}")

    elapsed = time.time() - t_start
    fps     = len(images) / elapsed if elapsed > 0 else 0
    print(f"\n{'='*55}")
    print(f"Images processed : {len(images)}")
    print(f"Total time       : {elapsed:.2f} s")
    print(f"Throughput       : {fps:.2f} fps")
    print(f"Violations found : {total_violations}")
    print(f"Results saved to : {output_folder}")
    print(f"{'='*55}")

    if show:
        cv2.destroyAllWindows()

    return {"n_images": len(images), "fps": fps, "violations": total_violations}


# ──────────────────────────────────────────────
# Video processing (Task B4)
# ──────────────────────────────────────────────

def process_video(video_path, output_path="output_video.mp4",
                  show=True, gt_violations=None):
    """
    Process a video file frame-by-frame.
    gt_violations: optional list of (start_frame, end_frame) ground-truth lane-change events.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    history_left  = deque(maxlen=Config.SMOOTH_FRAMES)
    history_right = deque(maxlen=Config.SMOOTH_FRAMES)

    # For evaluation metrics
    detected_violation_frames   = []
    consecutive_violation       = 0
    min_consecutive_for_event   = 8   # need N frames to call it a "lane change"
    events_detected             = []
    in_event                    = False
    event_start                 = 0

    frame_idx  = 0
    t_start    = time.time()
    roi_mask   = None

    print(f"Processing video: {video_path}  ({total} frames @ {fps_in:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi_mask is None:
            roi_mask = get_roi_mask(frame.shape)

        annotated, edges_bgr, violation, offset = process_frame(
            frame, roi_mask, history_left, history_right
        )

        # Event detection logic
        if violation:
            consecutive_violation += 1
            if consecutive_violation >= min_consecutive_for_event and not in_event:
                in_event    = True
                event_start = frame_idx - consecutive_violation + 1
        else:
            if in_event:
                events_detected.append((event_start, frame_idx - 1))
                in_event = False
            consecutive_violation = 0

        if violation:
            detected_violation_frames.append(frame_idx)

        # Overlay frame index + fps
        elapsed = time.time() - t_start
        live_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"Frame {frame_idx+1}/{total} | {live_fps:.1f} fps",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        writer.write(annotated)

        if show:
            display = cv2.resize(annotated, (960, 540))
            cv2.imshow("Lane Violation Detection", display)
            if cv2.waitKey(1) == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} | {live_fps:.1f} fps | events: {len(events_detected)}")

    # Finalize last event
    if in_event:
        events_detected.append((event_start, frame_idx - 1))

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0

    print(f"\n{'='*55}")
    print(f"Frames processed : {frame_idx}")
    print(f"Processing time  : {elapsed:.2f} s")
    print(f"Avg throughput   : {avg_fps:.2f} fps")
    print(f"Lane-change events detected: {len(events_detected)}")
    for i, (s, e) in enumerate(events_detected, 1):
        ts = s / fps_in
        te = e / fps_in
        print(f"  Event {i}: frames {s}-{e}  ({ts:.1f}s – {te:.1f}s)")

    # Evaluation against ground truth
    if gt_violations:
        evaluate(events_detected, gt_violations, frame_idx)

    print(f"Output saved to  : {output_path}")
    print(f"{'='*55}")

    return {
        "frames": frame_idx, "fps": avg_fps,
        "events": events_detected,
        "violation_frames": detected_violation_frames,
    }


# ──────────────────────────────────────────────
# Evaluation (Task B4)
# ──────────────────────────────────────────────

def frames_overlap(a_start, a_end, b_start, b_end):
    """Return True if two intervals overlap."""
    return a_start <= b_end and b_start <= a_end


def evaluate(detected, ground_truth, total_frames):
    """Print TP / FP / FN and accuracy statistics."""
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for d_s, d_e in detected:
        matched = False
        for i, (g_s, g_e) in enumerate(ground_truth):
            if frames_overlap(d_s, d_e, g_s, g_e):
                if i not in matched_gt:
                    tp += 1
                    matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'─'*40}")
    print("  Evaluation Results")
    print(f"{'─'*40}")
    print(f"  True  Positives (TP) : {tp}")
    print(f"  False Positives (FP) : {fp}  ← false detections")
    print(f"  False Negatives (FN) : {fn}  ← missed lane changes")
    print(f"  Precision            : {precision:.2%}")
    print(f"  Recall               : {recall:.2%}")
    print(f"  F1 Score             : {f1:.2%}")
    print(f"{'─'*40}")


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Road Lane Violation Detection (Part B)"
    )
    sub = parser.add_subparsers(dest="mode")

    # Image folder mode
    p_img = sub.add_parser("images", help="Process a folder of images")
    p_img.add_argument("folder", help="Path to image folder")
    p_img.add_argument("--output", default="output_images")
    p_img.add_argument("--no-show", action="store_true")

    # Video mode
    p_vid = sub.add_parser("video", help="Process a video file")
    p_vid.add_argument("video", help="Path to video file")
    p_vid.add_argument("--output", default="output_video.mp4")
    p_vid.add_argument("--no-show", action="store_true")
    p_vid.add_argument(
        "--gt", nargs="*", type=int,
        help="Ground-truth lane change events as pairs: start1 end1 start2 end2 ..."
    )

    args = parser.parse_args()

    if args.mode == "images":
        process_image_folder(args.folder, args.output, show=not args.no_show)

    elif args.mode == "video":
        gt = None
        if args.gt and len(args.gt) % 2 == 0:
            gt = [(args.gt[i], args.gt[i+1]) for i in range(0, len(args.gt), 2)]
        process_video(args.video, args.output, show=not args.no_show, gt_violations=gt)

    else:
        parser.print_help()
