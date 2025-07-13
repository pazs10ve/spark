# Real-Time Facial Emotion Detection

This repository contains a **real-time facial expression recognition** pipeline built on top of MediaPipe Face Detection, DeepFace emotion analysis and OpenCV visualisation utilities.

The project can be used in two different modes:

1. **Command-line / desktop preview** – analyse any local video file and preview the annotated frames in an OpenCV window.
2. **(Optional) Flask API / Stream** – serve frames to a web-based client (not included here) for browser-based visualisation.

---

## Features

* Detects faces in every frame using MediaPipe and draws bounding boxes.
* Classifies **7 emotions** (`angry, disgust, fear, happy, neutral, sad, surprise`) for every detected face via DeepFace.
* Generates a *polar graph* for each person to visualise per-emotion confidence distribution.
* Concatenates the graph(s) next to the original frame so that both raw footage and statistics are visible simultaneously.
* Logs frame-level emotion probabilities to JSON (`./uploads/data.json`) for downstream analysis.

---

## Quick Start

### 1. Clone and set-up environment

```bash
# clone the repo
$ git clone https://github.com/<YOUR_USERNAME>/spark.git
$ cd spark

# (Recommended) create virtualenv
$ python -m venv .venv
$ .venv/Scripts/activate            # Windows
$ source .venv/bin/activate         # macOS / Linux

# install python dependencies
$ pip install -r requirements.txt
```

Python 3.9+ is recommended.  All required packages and their exact versions are pinned in `requirements.txt`.

### 2. Download / prepare a video

Place the video you want to analyse in the project root.  Two sample files – `video.mp4` and a shorter `test.mp4` – are already provided.

### 3. Run the inference script

```bash
# use default test.mp4
$ python main.py                         

# or explicitly specify any file and window size
$ python main.py --video my_clip.mp4 --width 1200 --height 400
```

The annotated frames (bounding boxes + emotion labels + polar graphs) will pop up in a window titled **“Emotion Detection”**.  Press **`q`** at any time to stop.

All per-frame probabilities are appended to `./uploads/data.json` in the following schema:

```json
{
  "1": [ {"happy": 95.2, "sad": 1.3, ...}, ... ],
  "2": [ {"neutral": 78.7, "fear": 9.1, ...}, ... ]
}
```

where each top-level key represents a *person ID* detected in the frame.

---

## Optional: Serving via Flask

If you wish to broadcast the processed frames to a browser instead of an OpenCV window you can wrap the generator logic from `main.py` in a Flask route that yields multipart JPEGs.  The required packages (`Flask >= 3`) are already included in `requirements.txt`.

A minimal example (not included in the repo) would look like:

```python
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

Feel free to adapt the codebase to your specific UI/UX needs.

---

## Repository Structure

```
├── customImageConcat.py   # Utility to stitch multiple emotion-graphs together
├── detectFace.py          # Face detection, emotion analysis & visualisation classes
├── main.py                # CLI entry point – real-time video inference
├── requirements.txt       # Locked dependency versions
├── test.mp4               # Short demo clip
├── video.mp4              # Full-length sample video
└── README.md              # Project documentation (this file)
```

---

## Troubleshooting

* **`Could not open video`** – ensure the path supplied via `--video` exists and is readable.
* **GPU acceleration** – DeepFace falls back to CPU if no compatible GPU / CUDA drivers are found.  Inference will still work, albeit slower.
* **Missing DLLs on Windows** – For OpenCV related DLL errors install *Visual C++ Redistributable* from Microsoft.
* **MediaPipe / protobuf issues** – Completely remove existing mediapipe/protobuf wheels and reinstall the versions specified.

---

## License

This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* [Google MediaPipe](https://mediapipe.dev)
* [DeepFace](https://github.com/serengil/deepface)
* [OpenCV](https://opencv.org)

---

Happy hacking! :rocket:
