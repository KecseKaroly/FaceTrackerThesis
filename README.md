# Face Tracking with Siamese Network Re-Identification

This repository contains the implementation developed for the Master's Thesis:  
**"Increasing Robustness of Face Trackers with Siamese Networks"**  
by Károly Dániel Kecse  
Johannes Kepler University Linz – 2025

## 🧠 Overview

This project extends the ByteTrack multi-object tracking framework by introducing:
- A face-specific detection module (RetinaFace)
- An appearance feature extractor (MobileFaceNet)
- A Siamese Network trained using contrastive loss
- A voting-based re-identification system for long-term occlusion recovery

The system was evaluated on two custom video scenarios (Office and Shopping Mall), and achieved improved identity preservation while maintaining real-time performance.

## 🧩 Pipeline Components

- **Detector:** [RetinaFace](https://github.com/deepinsight/insightface)
- **Tracker:** Modified ByteTrack (Kalman Filter + Hungarian Matching)
- **Appearance Embedding:** MobileFaceNet
- **Re-ID Module:** Custom Siamese Network with contrastive loss and dynamic majority voting

## 📂 Folder Structure

```bash
.
├── tools/               # Main models (face_detector.py, embedding_extractor.py, siam_network.py) and main file (demo_track.py) which contain the main contributions
├── videos/              # Input videos 
├── logs/                # Output tracking logs (frame_id, track_id, bbox, score, etc.)
├── utils/               # Helper functions (timer and bounding box)
├── yoloy/tracker        # Main algorithms for tracker, kalman filter and the matching
└── requirements.txt
```
## 🧠 Overview

## Prerequisites

- **Python 3.11**
- CUDA installed on the system
- CUDA compatible tensorflow!
    -- Check CUDA version by: nvcc --version
    -- I had CUDA version 12.8
    -- Use  ***"pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"*** and modify it according to the version number
- ***install packages from the requirements.txt***




Run the code by: 
***python tools/demo_track.py video --path videos/office.mp4 --save_result***

Outputs are placed in the outputs/bytetrack_results/track_vis folder,
- the video_name.mp4 contains the whole output of the tracking
- the video_name2.mp4 shows the bounding boxes and their confidence scores
- demoFile.txt contains the main evens happening per frame: track created/lost/reactivated/removed
- demoFile2.txt contains the detailed results of the matching process, where Match N indicates the index of the matching phase
- the "matches" and "nonmatches" folder contains the outputs of the siamese process using identity@frame_number naming:

    - in "nonmatches", each folder contains an image of the detection which rejected the images of the current track, and we can see their distances
    - "matches" also contain images of detections and track images in a similar way, along with their pairwise distances
