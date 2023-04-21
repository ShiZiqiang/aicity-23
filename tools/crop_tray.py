from cv2 import line
from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import os
import numpy as np
from collections import Counter
import time
import sys
# from .deep_sort.deep_sort_app import *
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from deep_sort.deep_sort_app import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test net')
    parser.add_argument('--input_folder', help='the frames path')
    parser.add_argument('--out_file', help='the dir to save results')
    parser.add_argument('--detector', help='detector path', default="checkpoints/detectors_cascade_rcnn.pth")

    args = parser.parse_args()
    if args.input_folder is None:
        raise ValueError('--input_folder is None')
    if args.out_file is None:
        raise ValueError('--out_file is None')  
    return args


def find_traywohand(model, frame):

    result = inference_detector(model, frame)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    if len(bbox_result[0]) == 0:
        return None
    # white tray
    if len(bbox_result[61]) == 0:
        return None
    tray_bboxes = np.vstack(bbox_result[61])
    index_trays = np.argsort(tray_bboxes[:, -1])
    # person
    person_bboxes = np.vstack(bbox_result[0])
    person_indices = person_bboxes[:, -1] > 0.1
    person_segms = segm_result[0]
    person_segms = np.stack(person_segms, axis=0)
    person_segms = person_segms[person_indices]      # 1*1080*1920
    person_bboxes = person_bboxes[person_indices]
    # find tray without hand
    if index_trays.size:
        index_tray = index_trays[-1]
        bbox_tray = tray_bboxes[index_tray]
        if bbox_tray[-1] > 0.5:
            w = bbox_tray[2] - bbox_tray[0]
            h = bbox_tray[3] - bbox_tray[1]
            if (400 < bbox_tray[0] < 900) and (200 < bbox_tray[1] < 600) and (1000 < bbox_tray[2] < 1500) and (600 < bbox_tray[3] < 1000) and (700*500 <w*h < 900 * 700):
                for person_segm in person_segms:
                    for i in range(int(bbox_tray[0]), int(bbox_tray[2])):
                        for j in range(int(bbox_tray[1]), int(bbox_tray[3])):
                            if person_segm[j][i]:
                                return None
                return bbox_tray
    return None


def crop(frame, tray):
    frame = frame[int(tray[1]):int(tray[3])][:][:]
    frame = np.transpose(frame, (1, 0, 2))
    frame = frame[int(tray[0]):int(tray[2])][:][:]       
    frame = np.transpose(frame, (1, 0, 2))

    return frame




# init
args = parse_args()
# pretrain detector
pretrain_config_file = 'mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py'
pretrain_checkpoint_file = 'checkpoints/detectors_htc_r50_1x_coco-329b1453.pth'
pretrain_detector = init_detector(pretrain_config_file, pretrain_checkpoint_file)

def process(video_id):
    # frame path
    # video_id = 3
    frame_path = './frames/%d'%(video_id)

    # 找一帧白色托盘无人手
    start_time = time.time()
    white_tray = None
    
    for fid in range(1, len(os.listdir(frame_path)) + 1):
        frame = os.path.join(frame_path, 'img_%06d.jpg'%(fid))
        frame = cv2.imread(frame)
        result = find_traywohand(pretrain_detector, frame)
        if result is not None:
            mask_frame = crop(frame, result)
            white_tray = result
            cv2.imwrite(os.path.join('./crop_tray/', 'img_%d_%06d.jpg'%(video_id,fid)), mask_frame)
    
    #print('Find time:', end_time - start_time)


    return


if __name__ == '__main__':

    frame_path = args.input_folder
    videos = os.listdir(frame_path)
    videos.sort(key=lambda x: int(x))
    for i in range(len(videos)):
        print(int(videos[i]))
        process(int(videos[i]))
