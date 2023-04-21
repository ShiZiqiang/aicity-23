# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import sys
sys.path.append('./')
from .application_util import preprocessing
from .application_util import visualization
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker


def gather_sequence_info(detections, class_id=None):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """

    detections = np.array(detections)
    groundtruth = None
    image_size = None

    min_frame_idx = int(detections[:, 1].min())
    max_frame_idx = int(detections[:, 1].max())
    video_id = int(detections[0][0])
    update_ms = None

    feature_dim = detections.shape[1] - 124 if detections is not None else 0
    seq_info = {
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "video_id": video_id
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 1].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        # bbox, confidence, feature = row[2:6], row[6], row[10:]
        confidence, bbox, classification, feature = row[2], row[3:7], row[7:124], row[-1280:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, classification))
    return detection_list


def run(detections, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, class_id=None):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """

    seq_info = gather_sequence_info(detections, class_id)
    # print('【dyh】帧号从0开始')
    seq_info['min_frame_idx'] = seq_info['min_frame_idx']
    seq_info['max_frame_idx'] = seq_info['max_frame_idx']

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    # tracker = Tracker(metric, max_iou_distance=0.5, max_age=30) # wjf修改
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                seq_info["video_id"], frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]] + track.classification.tolist())

    # Run tracker.
    frame_interval = 1
    # print('【dyh】帧间隔：%d' % frame_interval)
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5, dyh_frame_interval=frame_interval)
    else:
        visualizer = visualization.NoVisualization(seq_info, dyh_frame_interval=frame_interval)
    visualizer.run(frame_callback)

    

    # Store results.
    # f = open('output_file', 'w')
    # array_wfeature = np.empty(shape=(0, 138))
    # for row in results:
    #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #         row[1] - 1, row[2], row[3], row[4], row[5], row[6]), file=f)
    np.save('output_file', np.array(results), allow_pickle=False)
    #     array_wfeature = np.concatenate(
    #         (
    #             array_wfeature,
    #             np.array([row[0] - 1, row[1], row[2], row[3], row[4], row[5], row[6], -1, -1, -1] + row[7].tolist())[np.newaxis, :]
    #         ),
    #         axis=0
    #     )
    # np.save(
    #     output_file.replace('.txt', '.npy'),
    #     array_wfeature,
    # )

    return results


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
