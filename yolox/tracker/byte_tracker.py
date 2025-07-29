import numpy as np
import os
import torch.nn as nn
import torch
import cv2
from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
import os.path as osp
import json
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import deque
from dataclasses import dataclass

@dataclass
class ScoredImage:
    image: any  # Replace `any` with your actual image type
    score: float

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, features, score, frame_id):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.confirmed = False
        self.features = features
        self.score = score
        self.tracklet_len = 0
        self.imgs = deque(maxlen=5)


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    
    def activate(self, kalman_filter, scoredImg, frame_id, save_folder):
        """Start a new tracklet"""
        self.imgs.append(scoredImg)
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        
        #self.is_activated = True
        if frame_id == 1:
            self.confirmed = True
            self.track_id = self.next_id()
            self.is_activated = True
            f = open(f"{save_folder}/demoFile.txt", 'a')
            f.write(f"[CONFIRMED]: {self.track_id} at {self.frame_id} with score: {self.score}\n")
            f.close()
            

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
    

    def update(self, new_track, frame_id, save_folder, scoredImg = None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.score = new_track.score

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        if scoredImg is not None:
            self.imgs.append(scoredImg)

        """important_pairs = {
                    (1, 2), (1, 245), (11, 463),
                    (2, 2), (2, 172), (9, 336),
                    (4, 135), (4, 555), (15, 733),
                    (5, 144), (5, 515), (12, 544), (12, 582), (17, 786),
                    (7, 171), (7, 406),(10, 445),
                    (13, 571), (13, 705), (16, 723),
                    (14, 642), (14, 867), (20, 912), 
                    (15, 677), (15, 726), (19, 863)
                }
        if (self.track_id, self.frame_id) in important_pairs:
            save_track_info(self, self.frame_id)"""

        if(not self.confirmed and self.tracklet_len > 1):
            self.track_id = self.next_id()
            self.confirmed = True
            f = open(f"{save_folder}/demoFile.txt", 'a')
            f.write(f"[CONFIRMED]: {self.track_id} at {self.frame_id} with score: {self.score}\n")
            f.close()
            
    
        

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


    

class BYTETracker():

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.args = args
        
        self.det_thresh = 0.27
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    

    def update(self, output_results, result_features, faces, img_info, img_size, siamese, save_folder):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if result_features.ndim == 3 and result_features.shape[0] == 1:
            result_features = result_features.squeeze(0)
            

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]
        
        #scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        #bboxes /= scale
        
        """ Most certain detections ABOVE THRESHOLD"""
        inds_first = scores > 0.7
        inds_low = scores > 0.28
        inds_between = scores < 0.7
        """ Less certain detections for second round of inference """
        inds_second = np.logical_and(inds_low, inds_between)
        dets_first = bboxes[inds_first]
        dets_second = bboxes[inds_second]
        scores_first = scores[inds_first]
        scores_second = scores[inds_second]
        features_first = result_features[inds_first] 
        features_second = result_features[inds_second]
        #faces_first = faces[inds_first]
        #faces_second = faces[inds_second]
        faces_first = [faces[i] for i, b in enumerate(inds_first) if b]
        faces_second = [faces[i] for i, b in enumerate(inds_second) if b]
        if len(dets_first) > 0:
            detections_first = [STrack(tlbr, f, s, self.frame_id) for (tlbr, f, s) in zip(dets_first, features_first, scores_first)]
        else:
            detections_first = []

        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''    
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        iou_dists = matching.iou_distance(strack_pool, detections_first)
        emb_dists = matching.embedding_distance(strack_pool, detections_first)
        
        alpha = 0.65
        fused_dists = alpha * iou_dists + (1 - alpha) * emb_dists
    
        
        matches, u_track, u_detection = matching.linear_assignment(fused_dists, thresh=0.662)

        for itracked, idet, match_score in matches:
            track = strack_pool[itracked]
            det = detections_first[idet]
            
            if track.features is not None and det.features is not None:
                alpha = 0.9
                track.features = alpha * track.features + (1 - alpha) * det.features
                track.features /= np.linalg.norm(track.features)

            if track.state == TrackState.Tracked:
                if(track.tracklet_len % 30 == 0 and det.score > 0.85):
                    track.update(detections_first[idet], self.frame_id, save_folder, ScoredImage(to_tensor(faces_first[idet]), track.score))
                else:
                    track.update(detections_first[idet], self.frame_id, save_folder)

                activated_starcks.append(track)
                f = open(f"{save_folder}/demoFile2.txt", 'a')
                f.write(f"[MATCH1]: {track.track_id} at {self.frame_id}  with score: {match_score}\n")
                f.close()
            else:
                track.re_activate(det, self.frame_id, False)
                f = open(f"{save_folder}/demoFile.txt", 'a')
                f.write(f"[REACTIVATE1]: {track.track_id} at {self.frame_id}  with match score: {match_score}\n")
                f.close()
                refind_stracks.append(track)

                
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(tlbr, f, s, self.frame_id) for (tlbr, f, s) in zip(dets_second, features_second, scores_second)]
        else:
            detections_second = []
        
        r_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked or strack_pool[i].state == TrackState.Lost]
        iou_dists = matching.iou_distance(r_stracks, detections_second)
        emb_dists = matching.embedding_distance(r_stracks, detections_second)
        alpha = 0.7
        fused_dists = alpha * iou_dists + (1 - alpha) * emb_dists
        matches, u_track, u_detection2 = matching.linear_assignment(fused_dists, thresh=0.68)
        
        for itracked, idet, match_score in matches:
            track = r_stracks[itracked]
            det = detections_second[idet]

            
            if track.features is not None and det.features is not None:
                alpha = 0.9
                track.features = alpha * track.features + (1 - alpha) * det.features
                track.features /= np.linalg.norm(track.features)

            if track.state == TrackState.Tracked:
                if(track.tracklet_len % 30 == 0):
                    track.update(det, self.frame_id, save_folder)
                else:
                    track.update(det, self.frame_id, save_folder)
                activated_starcks.append(track)
                f = open(f"{save_folder}/demoFile2.txt", 'a')
                f.write(f"[MATCH2]: {track.track_id} at {self.frame_id}  with match score: {match_score}\n")
                f.close()
            else:
                f = open(f"{save_folder}/demoFile.txt", 'a')
                f.write(f"[REACTIVATE2]: {track.track_id} at {self.frame_id} with match score: {match_score}\n")
                f.close()
                track.re_activate(det, self.frame_id, False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = r_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                f = open(f"{save_folder}/demoFile.txt", 'a')
                f.write(f"[LOST]: {track.track_id} at {self.frame_id} with score: {track.score}\n")
                f.close()
                lost_stracks.append(track)
        
        detections1 = [detections_first[i] for i in u_detection]
        detections2 = [detections_second[i] for i in u_detection2]
        detections = detections1 + detections2
        faces1 = [faces_first[i] for i in u_detection]
        faces2 = [faces_second[i] for i in u_detection2]
        faces = faces1 + faces2
        faces = [to_tensor(face) for face in faces]

        lost_tracks = [r_stracks[i] for i in u_track if r_stracks[i].state == TrackState.Lost]
        detections = majority_vote_matching(faces, detections, lost_tracks, self, siamese, refind_stracks, save_folder)
        #detections = average_vote_matching(faces, detections, lost_tracks, self, siamese, refind_stracks, save_folder)
        #detections = weighted_vote_matching(faces, detections, lost_tracks, self, siamese, refind_stracks, save_folder)

        iou_dists = matching.iou_distance(unconfirmed, detections)
        emb_dists = matching.embedding_distance(unconfirmed, detections)
        
        alpha = 0.5
        fused_dists = alpha * iou_dists + (1 - alpha) * emb_dists
        fused_dists = matching.fuse_score(fused_dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(fused_dists, thresh=self.args.match_thresh)
        
        for itracked, idet, match_score in matches:
            f = open(f"{save_folder}/demoFile2.txt", 'a')
            f.write(f"[MATCH3]: {track.track_id} at {self.frame_id}  with score: {match_score}\n")
            f.close()
            if unconfirmed[itracked].features is not None and detections[idet].features is not None:
                alpha = 0.9 
                cos_sim = np.dot(unconfirmed[itracked].features, detections[idet].features)
                if cos_sim > 0.6:
                    unconfirmed[itracked].features = alpha * unconfirmed[itracked].features + (1 - alpha) * detections[idet].features
                    unconfirmed[itracked].features /= np.linalg.norm(unconfirmed[itracked].features)
            unconfirmed[itracked].update(detections[idet], self.frame_id, save_folder)
            activated_starcks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < 0.1:
                continue
            img = faces[inew]

            
            track.activate(self.kalman_filter, ScoredImage(img, track.score), self.frame_id, save_folder)
            activated_starcks.append(track)
            f = open(f"{save_folder}/demoFile.txt", 'a')
            f.write(f"[NEW] detection at {self.frame_id} with score: {track.score} > {self.det_thresh}\n")
            f.close()
            

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

                f = open(f"{save_folder}/demoFile.txt", 'a')
                if(track.confirmed):
                    f.write(f"[2REMOVED] confirmed track with id {track.track_id} at {self.frame_id}\n")
                else:
                    f.write(f"[2REMOVED] unconfirmed track with id {track.track_id} at {self.frame_id}\n")
                f.close()


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def save_track_info(track, frame_id):
    # Create the folder if it doesn't exist
    os.makedirs("test_faces", exist_ok=True)

    # Save the track's image as a .jpg
    track_image = to_np(track.img)
    track_image_path = f"test_faces/{track.track_id}_{frame_id}.jpg"
    cv2.imwrite(track_image_path, track_image)

def to_tensor(img):
    if isinstance(img, np.ndarray):
        if img.dtype == object:
            try:
                img = img.astype(np.uint8)
            except Exception as e:
                raise ValueError(f"Failed to convert image to uint8: {e}")
        
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0).to("cuda")

def to_np(img):
    if isinstance(img, torch.Tensor):
        
        img = img.squeeze(0)  
        img = img.permute(1, 2, 0).cpu().numpy() 
        img = (img * 255).astype(np.uint8) 
    return img


def majority_vote_matching(faces, detections, lost_tracks, self, siamese, refind_stracks, save_folder):
    unmatched_detections = []

    for i, unmatched_face in enumerate(faces):
        corresponding_det = detections[i]
        matched = False

        for lost_track in lost_tracks[::-1]:
            match_count = 0
            dists = []
            for scored_img in lost_track.imgs:
                output1, output2 = siamese(scored_img.image, unmatched_face)
                distance = nn.functional.pairwise_distance(output1, output2).item()
                if distance < 0.4357:
                    match_count += 1
                dists.append(distance)
                

            if match_count >= 3:
                print(f"[MAJORITY MATCH] Frame {self.frame_id}, Lost Track {lost_track.track_id}")
                if not os.path.isdir(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}"):
                    os.makedirs(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/", exist_ok=True)
                cv2.imwrite(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/{i}_det.jpg", to_np(unmatched_face))
                for i, scoredImg in enumerate(lost_track.imgs):
                    cv2.imwrite(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/{i}_{dists[i]}_track.jpg", to_np(scoredImg.image))
                lost_track.re_activate(corresponding_det, self.frame_id, False)
                refind_stracks.append(lost_track)
                with open(f"{save_folder}/demoFile.txt", 'a') as f:
                    f.write(f"[REACTIVATE_MAJORITY]: {lost_track.track_id} at {self.frame_id}, with {match_count} votes\n")
                matched = True
                break
            else:
                if not os.path.isdir(f"{save_folder}/nonmatches/{lost_track.track_id}@{self.frame_id}/"):
                    os.makedirs(f"{save_folder}/nonmatches/{lost_track.track_id}@{self.frame_id}", exist_ok=True)
                cv2.imwrite(f"{save_folder}/nonmatches/{lost_track.track_id}@{self.frame_id}/{i}_det.jpg", to_np(unmatched_face))
                for i, scoredImg in enumerate(lost_track.imgs):
                    cv2.imwrite(f"{save_folder}/nonmatches/{lost_track.track_id}@{self.frame_id}/{i}_{dists[i]}_track.jpg", to_np(scoredImg.image))
                

        if not matched:
            unmatched_detections.append(corresponding_det)

    return unmatched_detections


def average_vote_matching(faces, detections, lost_tracks, self, siamese, refind_stracks, save_folder):
    unmatched_detections = []

    for i, unmatched_face in enumerate(faces):
        corresponding_det = detections[i]
        matched = False

        for lost_track in lost_tracks[::-1]:
            distances = []

            for scored_img in lost_track.imgs:
                output1, output2 = siamese(scored_img.image, unmatched_face)
                distance = nn.functional.pairwise_distance(output1, output2).item()
                distances.append(distance)

            avg_distance = sum(distances) / len(distances)
            if avg_distance < 0.47:
                print(f"[AVG MATCH] Frame {self.frame_id}, Lost Track {lost_track.track_id}, Avg Dist: {avg_distance:.3f}")
                if not os.path.isdir(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}"):
                    #os.mkdir(f"{save_folder}/matches/")
                    os.mkdir(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/")
                cv2.imwrite(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/{self.frame_id}_{lost_track.track_id}_det.jpg", to_np(unmatched_face))
                for i, scoredImg in enumerate(lost_track.imgs):
                    cv2.imwrite(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/{self.frame_id}_{lost_track.track_id}_{i}_track.jpg", to_np(scoredImg.image))
                lost_track.re_activate(corresponding_det, self.frame_id, False)
                refind_stracks.append(lost_track)
                with open(f"{save_folder}/demoFile.txt", 'a') as f:
                    f.write(f"[REACTIVATE_AVG]: {lost_track.track_id} at {self.frame_id} (avg dist: {avg_distance:.3f}), used faces: {len(lost_track.imgs)}\n")
                matched = True
                break
            # else:
            #     print(f"{self.frame_id}: {avg_distance} for {lost_track.track_id}")

        if not matched:
            unmatched_detections.append(corresponding_det)

    return unmatched_detections

def weighted_vote_matching(faces, detections, lost_tracks, self, siamese, refind_stracks, save_folder):
    unmatched_detections = []

    for i, unmatched_face in enumerate(faces):
        corresponding_det = detections[i]
        matched = False

        for lost_track in lost_tracks[::-1]:
            weighted_sum = 0.0
            total_weight = 0.0

            for scored_img in lost_track.imgs:
                output1, output2 = siamese(scored_img.image, unmatched_face)
                distance = nn.functional.pairwise_distance(output1, output2).item()
                weight = scored_img.score
                weighted_sum += distance * weight
                total_weight += weight

            if total_weight == 0:
                continue

            weighted_avg = weighted_sum / total_weight
            if weighted_avg < 0.47:
                print(f"[WEIGHTED MATCH] Frame {self.frame_id}, Lost Track {lost_track.track_id}, Weighted Dist: {weighted_avg:.3f}")
                if not os.path.isdir(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}"):
                    os.mkdir(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}")
                cv2.imwrite(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/{self.frame_id}_{lost_track.track_id}_det.jpg", to_np(unmatched_face))
                for i, scoredImg in enumerate(lost_track.imgs):
                    cv2.imwrite(f"{save_folder}/matches/{lost_track.track_id}@{self.frame_id}/{self.frame_id}_{lost_track.track_id}_{i}_track.jpg", to_np(scoredImg.image))
                lost_track.re_activate(corresponding_det, self.frame_id, False)
                refind_stracks.append(lost_track)
                with open(f"{save_folder}/demoFile.txt", 'a') as f:
                    f.write(f"[REACTIVATE_WEIGHTED]: {lost_track.track_id} at {self.frame_id} (weighted dist: {weighted_avg:.3f}), used faces: {len(lost_track.imgs)}\n")
                matched = True
                break

        if not matched:
            unmatched_detections.append(corresponding_det)

    return unmatched_detections