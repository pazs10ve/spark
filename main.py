"""
process.py
-----------
Standalone script to perform real-time emotion detection inference on a local
video file (default: "video.mp4").

The script replicates the inference logic used in `main.py`'s `gen` generator
but runs it locally, rendering the annotated frames in a cv2 window instead of
streaming them through Flask.

• Draws face bounding boxes and emotion labels for each detected face.
• Generates polar-graph visualisations for every frame (one per detected face)
  and concatenates the graph(s) next to the original frame using the existing
  `custom_concat` helper.
• Press the "q" key to terminate processing early.

Usage (from repo root):
    python -m app.process            # uses default video.mp4 in repo root
    python -m app.process --video myclip.mp4 --width 1200 --height 400

Requirements: identical to those already listed for the Flask application.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .detectFace import FaceDetect, EmotionAnalyze, emotionVisualize
from .customImageConcat import custom_concat


face = FaceDetect()
emotion = EmotionAnalyze()
visualize = emotionVisualize()


def process_video(
    video_path: str = "test.mp4",
    *,
    width: int = 1200,
    height: int = 400,
    data_path: str = "./uploads/data.json",
    display_window: str = "Emotion Detection",
):
    """Run emotion detection on *video_path* and display results live."""

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Ensure output directory exists for json data logging
    os.makedirs(Path(data_path).parent, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Video finished

        # ---------------- Emotion inference & visualisation -----------------
        frame = face.drawBbox(frame)
        frame, emots = emotion.labelEmotion(frame)
        frame = cv2.resize(frame, (700, 500))

        if emots:
            graphs = []
            num_persons = len(emots)
            for person_idx, emot_probs in emots.items():
                visualize.visualize(person_idx, emot_probs)
                graph = visualize.plane
                # Resize graph so that total width remains manageable
                g_h, g_w = graph.shape[:2]
                graph = cv2.resize(
                    graph,
                    (
                        int(g_w / max(num_persons, 1)),
                        int(g_h / max(num_persons, 1)),
                    ),
                )
                graphs.append(graph)

                # -------------------------- Logging -------------------------
                # Persist probabilities for each person in a JSON file.
                if not os.path.exists(data_path):
                    with open(data_path, "w", encoding="utf-8") as f:
                        json.dump({}, f, indent=4)

                with open(data_path, "r", encoding="utf-8") as f:
                    records = json.load(f)

                key = str(person_idx + 1)
                records.setdefault(key, []).append(emot_probs)

                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=4)

            # Concatenate emotion graphs vertically / horizontally depending
            # on face count (uses existing helper for up to 10 faces).
            emot_graph = (
                custom_concat(*graphs) if num_persons > 1 else graphs[0]
            )
        else:  # No faces detected in frame
            visualize.visualize(
                "unknown",
                dict.fromkeys(list(visualize.line_angles.keys())[:-1], 0),
            )
            emot_graph = visualize.plane

        # Combine original frame with emotion graph(s)
        emot_graph = cv2.resize(emot_graph, (500, 500))
        frame_combined = np.concatenate((frame, emot_graph), axis=1)
        frame_combined = cv2.resize(frame_combined, (width, height))

        # --------------------------- Display --------------------------------
        cv2.imshow(display_window, frame_combined)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Optional: throttle playback slightly to mimic ~14 fps
        # time.sleep(0.07)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------- #
# CLI entry point                                                              #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time emotion detection on video files.")
    parser.add_argument("--video", dest="video", default="video.mp4", help="Path to input video file.")
    parser.add_argument("--width", dest="width", type=int, default=1200, help="Display window width.")
    parser.add_argument("--height", dest="height", type=int, default=400, help="Display window height.")
    parser.add_argument(
        "--data", dest="data", default="./uploads/data.json", help="Path to JSON file for storing per-frame emotion data.")
    args = parser.parse_args()

    process_video(args.video, width=args.width, height=args.height, data_path=args.data)
