import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np

from loguru import logger
from yolox.utils.visualize import plot_tracking, plot_tracking2
import importlib
import yolox.tracker.byte_tracker
importlib.reload(yolox.tracker.byte_tracker)
from yolox.tracker.byte_tracker import BYTETracker

from yolox.tracking_utils.timer import Timer

from embedding_extractor import EmbeddingExtractor
from facedetector import FaceDetector

from data import cfg_mnet

from siam_network import SiameseNetwork




IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--path", default="./videos/office.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_buffer", type=int, default=220, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.65, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=5.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=3, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def imageflow_demo(detector, extractor, siamese, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
        file, base = (args.path.split("/")[-1]).split(".")
        save_path2 = osp.join(save_folder, (f"{file}_2.{base}"))
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    vid_writer2 = cv2.VideoWriter(
        save_path2, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    tracking_results = [] 

    f = open(f"{save_folder}/demoFile.txt", 'w')
    f.write("")
    f.close()
    f = open(f"{save_folder}/demoFile2.txt", 'w')
    f.write("")
    f.close()
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ret_val, frame = cap.read()
        if not ret_val:
            print("⚠️ Error: Unable to read the video. Check file path or codec.")
            break
        h, w, _ = frame.shape
        dets, img_info = detector.detect_faces(frame, timer)
        if not dets.any():
            continue
        cropped_faces = []
        xywh_dets = []
        for box in dets:
                x1, y1, x2, y2, score = box[:5]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                cropped_face = frame[int(y1):int(y2), int(x1):int(x2)]
                xywh_dets.append([x1, y1, x2-x1, y2-y1, score]) 
                cropped_faces.append(cropped_face)
        det_features = extractor.extract(cropped_faces)
    
        xywh_dets = np.array(xywh_dets)
        
        #cropped_faces = np.asarray(cropped_faces, dtype="object")
        if not xywh_dets.any():
            continue
        online_targets = tracker.update(xywh_dets, det_features, cropped_faces, [img_info['height'], img_info['width']], [width, height], siamese, save_folder)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        for t in online_targets:
            if t.track_id == 0:
                continue
            x, y, w, h = map(int, t.tlwh)
            tid = t.track_id
            score = t.score

            online_tlwhs.append(t.tlwh)
            online_ids.append(tid)
            online_scores.append(score)
            tracking_results.append(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f},-1,-1,-1\n")
        
        timer.toc()
        online_im = plot_tracking(
            img_info['raw_img'], online_tlwhs, online_ids, online_scores, frame_id=frame_id + 1, fps=1. / timer.average_time
        )
        vid_writer.write(online_im)
        
        online_im2 = plot_tracking2(
            img_info['raw_img'], xywh_dets, frame_id=frame_id + 1, fps=1. / timer.average_time
        )
        vid_writer2.write(online_im2)
        
        frame_id += 1

def main(args):
    output_dir = osp.join("./outputs", "bytetrack_results")
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    trained_model_path = './pretrained/mobilenet0.25_Final.pth'
    cfg = cfg_mnet
    detector = FaceDetector(cfg, trained_model_path)

    extractor = EmbeddingExtractor(model_path = "./pretrained/MobileFaceNet.tflite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siamese = SiameseNetwork().to(device)
    siamese.load_state_dict(torch.load("./pretrained/siamese_model_64-64.pth", map_location=device, weights_only=True))
    current_time = time.localtime()
    
    imageflow_demo(detector, extractor, siamese, vis_folder, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    
    main(args)