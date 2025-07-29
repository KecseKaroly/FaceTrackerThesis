#!/bin/bash

: '
docker run --gpus all -it --rm ^
-v D:/Thesis/ByteTrack/pretrained:/workspace/ByteTrack/pretrained ^
-v D:/Thesis/ByteTrack/datasets:/workspace/ByteTrack/datasets ^
-v D:/Thesis/ByteTrack/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs ^
--net=host ^
--privileged ^
ubuntu:latest /bin/bash
'

docker run --gpus all -it --rm \
  -v D:/Thesis/ByteTrack/pretrained:/workspace/ByteTrack/pretrained \
  -v D:/Thesis/ByteTrack/datasets:/workspace/ByteTrack/datasets \
  -v D:/Thesis/ByteTrack/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
  --net=host \
  --privileged \
  ubuntu:latest /bin/bash
