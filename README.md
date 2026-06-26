# Real-Time Lane Violation Detection System

A Computer Vision based real-time road lane violation detection system built using Python and OpenCV. The system detects lane boundaries from video streams and identifies lane violations using image processing and geometric analysis techniques.

## Features

* Real-time lane detection from video input
* Canny Edge Detection for feature extraction
* Hough Line Transform for lane line detection
* Region of Interest (ROI) masking
* Lane line averaging and temporal smoothing across frames
* Offset-based lane violation detection
* Works on both image sequences and videos
* Performance evaluation using:

  * Precision
  * Recall
  * F1-Score

## Technologies Used

* Python
* OpenCV
* NumPy

## System Pipeline

1. Frame Extraction from Video
2. Grayscale Conversion
3. Gaussian Blur
4. Canny Edge Detection
5. ROI Masking
6. Hough Line Transform
7. Lane Line Averaging
8. Temporal Smoothing
9. Lane Violation Detection
10. Performance Evaluation

## Project Structure

```bash
├── input_videos/
├── output_videos/
├── images/
├── results/
├── main.py
├── utils.py
├── evaluation.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/real-time-lane-violation-detection.git
cd real-time-lane-violation-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the detection system:

```bash
python main.py
```

For custom video input:

```bash
python main.py --video path/to/video.mp4
```

## Evaluation Metrics

The system performance is evaluated using:

* **Precision** — Correct violation detections out of all predicted violations
* **Recall** — Correct violation detections out of all actual violations
* **F1-Score** — Harmonic mean of Precision and Recall

## Results

The system successfully detects lane boundaries and identifies lane violations in real time with stable frame-to-frame tracking using temporal smoothing techniques.

## Future Improvements

* Deep Learning based lane segmentation
* Multi-lane tracking
* Vehicle detection integration
* GPU acceleration for faster processing
* Night-time and adverse weather handling

## Applications

* Smart Traffic Monitoring
* Road Safety Systems
* Intelligent Transportation Systems (ITS)
* Automated Traffic Surveillance
