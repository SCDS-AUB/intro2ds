---
layout: default
title: "DATA 202 Module 6: Video Data and Action Recognition"
---

# DATA 202 Module 6: Video Data and Action Recognition

## Introduction

Video is the richest data modality—combining visual information across time. YouTube hosts over 800 million videos; TikTok users upload millions more daily; surveillance systems record continuously. Processing video data requires understanding not just what appears in frames but how things change across time.

This module explores video analysis: from basic processing to action recognition, object tracking, and video understanding with deep learning.

---

## Part 1: Video Fundamentals

### What is Video?

Video is a sequence of images (frames) displayed rapidly to create the illusion of motion:
- **Frame rate**: Frames per second (24 fps for film, 30 fps for TV, 60+ for games)
- **Resolution**: Pixels per frame (1920×1080 for HD, 3840×2160 for 4K)
- **Bit rate**: Data per second of video
- **Codec**: Compression algorithm (H.264, H.265, VP9, AV1)

One minute of uncompressed 4K video at 30 fps:
- 3840 × 2160 pixels × 3 bytes × 30 fps × 60 seconds ≈ 44 GB

Compression is essential.

### Video as Structured Data

Extract structured information from video:
- **Per-frame**: Objects, faces, text
- **Temporal**: Motion, actions, events
- **Aggregated**: Statistics, summaries, highlights

### Video Processing Pipeline

```python
import cv2

# Read video
cap = cv2.VideoCapture('video.mp4')

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display or save results
    cv2.imshow('Frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

---

## Part 2: Object Detection and Tracking

### Detection in Video

Apply image object detection (YOLO, Faster R-CNN) to each frame:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
```

### Object Tracking

**Challenge**: Link detections across frames to track objects.

**Approaches**:
- **SORT (Simple Online Realtime Tracking)**: Kalman filter + Hungarian algorithm
- **DeepSORT**: Add appearance features for re-identification
- **ByteTrack**: Consider low-confidence detections
- **Transformers**: MOTR, TrackFormer

```python
# Simple tracking with DeepSORT
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

# For each frame
detections = detect(frame)  # Get detections
tracks = tracker.update_tracks(detections, frame=frame)

for track in tracks:
    if track.is_confirmed():
        track_id = track.track_id
        bbox = track.to_ltrb()
```

---

## Part 3: Action Recognition

### Understanding Actions in Video

**Action recognition** classifies what activity is occurring:
- Running, jumping, waving
- Cooking, playing piano
- Fighting, stealing (for security)

**Approaches**:
1. **Two-Stream**: Separate spatial (appearance) and temporal (motion) streams
2. **3D Convolutions**: Extend CNNs to space-time (C3D, I3D)
3. **Temporal Modeling**: Use recurrent networks on frame features
4. **Transformers**: Video Transformers (ViViT, TimeSformer)

### Optical Flow

**Optical flow** captures motion between frames—the apparent motion of pixels.

```python
import cv2

# Lucas-Kanade optical flow
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)

# Dense optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
```

### Video Transformers

Modern video understanding uses Transformers:
- Patch frames into tokens
- Add temporal position encoding
- Self-attention across space and time

**Models**: ViViT, TimeSformer, Video Swin Transformer

---

## Part 4: Video Understanding at Scale

### Video Captioning

Generate text descriptions of video content:
- "A man is playing guitar in a park"
- "The cat jumps onto the table and knocks over a glass"

### Video Question Answering

Answer questions about video:
- "What color is the car that appears first?"
- "How many people enter the room?"

### Video-Language Models

**Multimodal models** align video with language:
- **CLIP for Video**: Extend image-text to video-text
- **VideoBERT**: BERT for video
- **Video-LLaMA, Video-ChatGPT**: LLMs that understand video

---

## DEEP DIVE: The Rise of TikTok and Short-Form Video AI

### The Algorithm That Learned What You Want

TikTok's recommendation algorithm became legendary for its accuracy—users report feeling "understood" within minutes of first use. How?

**Key Components**:
1. **Video Understanding**: Deep learning extracts visual and audio features
2. **Engagement Signals**: Watch time, replays, shares, comments
3. **User Modeling**: Build preference profiles from behavior
4. **Real-Time Learning**: Adapt quickly to changing interests

**The For You Page** algorithm reportedly:
- Uses hundreds of signals per video
- Weighs watch completion heavily
- Diversifies to avoid filter bubbles (somewhat)
- Serves new content to test user response

### The Dark Side

Optimization for engagement creates problems:
- Addictive feedback loops
- Misinformation spread
- Mental health impacts
- Polarization concerns

Video recommendation is a powerful demonstration of AI's ability to shape behavior—for better and worse.

---

## HANDS-ON EXERCISE: Video Analysis Pipeline

### Part 1: Object Detection in Video

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detections_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            detections_log.append({
                'frame': frame_count,
                'class': model.names[cls],
                'confidence': conf
            })

        annotated = results[0].plot()
        out.write(annotated)
        frame_count += 1

    cap.release()
    out.release()
    return pd.DataFrame(detections_log)
```

### Part 2: Motion Analysis

```python
def compute_motion_magnitude(video_path, sample_rate=5):
    """Compute average motion magnitude per sampled frame."""
    cap = cv2.VideoCapture(video_path)
    motion_data = []
    prev_gray = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_data.append({
                    'frame': frame_idx,
                    'mean_motion': np.mean(magnitude),
                    'max_motion': np.max(magnitude)
                })

            prev_gray = gray

        frame_idx += 1

    cap.release()
    return pd.DataFrame(motion_data)
```

---

## Recommended Resources

### Libraries
- **OpenCV**: Video processing fundamentals
- **Ultralytics YOLO**: Object detection
- **MMAction2**: Action recognition toolkit
- **PyTorchVideo**: Facebook's video understanding library

### Datasets
- **Kinetics**: Large-scale action recognition
- **ActivityNet**: Untrimmed video understanding
- **YouTube-8M**: Video classification
- **MOT Challenge**: Multi-object tracking

---

*Module 6 explores video data—the richest but most computationally demanding data modality. From object tracking to action recognition to video understanding, we learn how machines interpret the temporal visual world.*
